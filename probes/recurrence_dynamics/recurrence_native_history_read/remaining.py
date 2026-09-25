"""Seven frozen native-read cells: bowl progress, then cow progress."""

from __future__ import annotations

import argparse
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
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.integrations import sdpa_attention

from probes.recurrence_dynamics.recurrence_native_history_read import run as first
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


UNIT = first.UNIT
ADMISSION = UNIT / "lead-native-read-remaining-admission-v1.json"
PREFLIGHT = UNIT / "supporting/native-read-remaining-preflight-v1.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing/native-read-remaining-v1")
ADMISSION_SHA = "c5e2c4b862bc3ffb3a4f954cb44bcd3dfeaaf3f63fbbf612bc71b2819f9539ca"
CONDITIONS = {"bowl-row2": ("native", "identity-mask-sham", "latest-row-mask", "earlier-row-mask"),
              "cow-row1": ("native", "identity-mask-sham", "latest-row-mask")}
GEOMETRY = {
    "bowl-row2": {"image": "mature:313465:0", "group": "fresh-18", "target": 3,
                  "query": (20, 25), "latest": (10, 20), "earlier": (0, 10),
                  "old": 151675, "new": 151827, "old0": 151670,
                  "lengths": [286, 109, 11, 138], "header": 5},
    "cow-row1": {"image": "mature:479075:0", "group": "fresh-26", "target": 3,
                 "query": (9, 13), "latest": (0, 9), "earlier": None,
                 "old": 151823, "new": 151980, "old0": None,
                 "lengths": [21, 20, 19, 100], "header": 4},
}
TOL = 2e-4


def require(value, message):
    first.require(value, message)


def binding(path):
    return first.bind(path)


def contract():
    require(binding(ADMISSION)["sha256"] == ADMISSION_SHA, "remaining admission changed")
    admission = json.loads(ADMISSION.read_text())
    require(admission["status"] == "lead-admitted-fixed-remaining-seven" and
            admission["max_additional_model_forwards"] == admission["max_additional_vision_forwards"] == 7
            and admission["max_free_tokens"] == 0 and
            admission["ownership"]["raw_output_root"] == str(OUT) and
            admission["panel_cap_gpu_seconds"] == 900 and
            math.isclose(admission["prior_panel_charge_gpu_seconds"], 20.247578292, abs_tol=1e-10) and
            math.isclose(admission["prior_sequence_gpu_hours"], .24381016975279973, abs_tol=1e-12) and
            admission["sequence_cap_gpu_hours"] == 8, "remaining scope/cost changed")
    for name in ("protocol", "first_case_acceptance", "first_case_verification", "candidate_registry",
                 "preserve_first_case_producer", "preserve_first_case_candidate", "cost_audit"):
        expected = admission[name]
        require(binding(expected["path"])["sha256"] == expected["sha256"], f"predecessor changed: {name}")
    accepted = json.loads(Path(admission["first_case_acceptance"]["path"]).read_text())
    for name in ("receipt", "readback"):
        require(binding(accepted[name]["path"])["sha256"] == accepted[name]["sha256"],
                f"accepted first-case raw changed: {name}")
    require(accepted["charged_gpu_seconds"] == admission["prior_panel_charge_gpu_seconds"] and
            accepted["cumulative_sequence_gpu_hours"] == admission["prior_sequence_gpu_hours"],
            "accepted prior charge changed")
    registry = json.loads(Path(admission["candidate_registry"]["path"]).read_text())
    expected_cells = []
    for state, conditions in CONDITIONS.items():
        g = GEOMETRY[state]
        for condition in conditions:
            key = g["earlier"] if condition == "earlier-row-mask" else g["latest"]
            expected_cells.append({"state": state, "condition": condition, "image": g["image"],
                                   "source_group": g["group"], "target_batch_index": g["target"],
                                   "current_raw_query_span": list(g["query"]),
                                   "selected_raw_key_span": list(key),
                                   "old_token_id": g["old"], "new_token_id": g["new"]})
    require(admission["admitted_cells"] == expected_cells, "seven cells/order changed")
    for state, g in GEOMETRY.items():
        image = registry["images"][g["image"]]
        require(image["group"] == g["group"] and image["batch_index"] == g["target"] and
                image["full_batch_target_lengths"] == g["lengths"] and
                image["model_identity"]["model"] == "untied",
                f"source identity/shape changed: {state}")
        for name in ("raw", "trace", "runtime_receipt", "image"):
            expected = image["source_bindings"][name]
            require(binding(expected["path"])["sha256"] == expected["sha256"],
                    f"source binding changed: {state}:{name}")
    return admission, registry


def source_inputs(model, batch, raw, g, pad):
    offset = g["query"][1]
    tails = _prefix_tokens(raw, offset, pad)
    histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
    full = exact_history_inputs(model, batch.inputs, histories, pad_token_id=pad, logits_to_keep=1)
    width = full["input_ids"].shape[1]
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    prompt = width - offset
    require(full["input_ids"].shape == full["attention_mask"].shape == (4, width) and
            full["position_ids"].shape == (3, 4, width) and
            full["input_ids"][:, prompt:].tolist() == tails and
            full["attention_mask"][:, prompt:].tolist() == [[1]*offset]*4 and
            len(raw) == 4 and [len(row["token_ids"]) for row in raw] == g["lengths"],
            "source full-prefix shape/companions changed")
    t = raw[g["target"]]["token_ids"]
    q0, q1 = g["query"]
    require(q1-q0 == g["header"] and t[q1] == g["new"] and
            full["input_ids"][g["target"], prompt+q0:prompt+q1].tolist() == t[q0:q1] and
            t[g["latest"][0]:g["latest"][1]][-1] == 151649,
            "native current header/first x1 or latest complete row changed")
    if g["earlier"] is not None:
        require(t[g["earlier"][0]:g["earlier"][1]][-1] == 151649 and
                t[g["earlier"][0]+5] == g["old0"] and
                t[g["latest"][0]+5] == g["old"], "bowl earlier/latest owner extents changed")
    else:
        require(t[g["latest"][0]+g["header"]] == g["old"], "cow old extent changed")
    return full, prompt


def prepare_source(q, image, g, device):
    b = image["source_bindings"]
    raw = json.loads(Path(b["raw"]["path"]).read_text())["rows"]
    boundary = {"group": g["group"], "batch_index": g["target"],
                "image_id": int(g["image"].split(":")[1]), "raw_path": b["raw"]["path"],
                "trace_path": b["trace"]["path"], "receipt_path": b["runtime_receipt"]["path"],
                "native_tokens": raw[g["target"]]["token_ids"],
                "native_token_hash": token_hash(raw[g["target"]]["token_ids"])}
    panel = json.loads((MATURE / "panel.json").read_text())
    batch, raw, trace, group, planning = _source(boundary, "untied", panel, q, device)
    receipt = json.loads(Path(b["runtime_receipt"]["path"]).read_text())
    require(len(raw) == len(group["cases"]) == 4 and
            input_identity(batch) == receipt["input_identity"] and
            list(batch.request_ids) == image["request_ids"], "original full batch identity changed")
    return batch, raw, trace, receipt, planning


def mask_for(full, prompt, g, condition, *, target=3, query=None, key=None, source_hash=None):
    expected_query = g["query"]
    expected_key = g["earlier"] if condition == "earlier-row-mask" else g["latest"]
    if query is None: query = expected_query
    if key is None: key = expected_key
    source = full["attention_mask"]
    require(condition in CONDITIONS[g["state"]] and target == g["target"] and
            query == expected_query and key == expected_key and
            prompt == source.shape[1]-expected_query[1] and source.ndim == 2 and
            (source_hash is None or tensor_hash(source) == source_hash),
            "mask caller target/query/key/source geometry changed")
    base = first.native_4d(source)
    if condition == "native": return source, base
    actual = base.clone()
    if condition in ("latest-row-mask", "earlier-row-mask"):
        actual[target, 0, prompt+query[0]:prompt+query[1], prompt+key[0]:prompt+key[1]] = False
    verify_mask(base, actual, prompt, g, condition)
    return actual, base


def verify_mask(base, actual, prompt, g, condition):
    key = g["earlier"] if condition == "earlier-row-mask" else g["latest"]
    q = g["query"]
    rect = (g["target"], 0, slice(prompt+q[0], prompt+q[1]), slice(prompt+key[0], prompt+key[1]))
    expected = base.clone()
    if condition in ("latest-row-mask", "earlier-row-mask"):
        require(torch.all(base[rect]).item(), "selected native row not readable")
        expected[rect] = False
    require(torch.equal(expected, actual), "selected/complement/companion mask changed")


def forward_cell(model, full, prompt, g, condition, observer=None, source_hash=None):
    mask, base = mask_for(full, prompt, g, condition, source_hash=source_hash)
    if observer is not None:
        observer["expected_mask"] = base if condition == "native" else mask
        observer["condition"] = condition
    return model(**{**full, "attention_mask": mask})


def cpu_fixture(full, prompt, g):
    source = full["attention_mask"]
    source_hash = tensor_hash(source)
    class Consumer:
        def __init__(self): self.masks = []
        def __call__(self, **kwargs):
            self.masks.append(kwargs["attention_mask"].clone())
            return kwargs["attention_mask"]
    fake = Consumer()
    for condition in CONDITIONS[g["state"]]:
        forward_cell(fake, full, prompt, g, condition, source_hash=source_hash)
    base = first.native_4d(source)
    require(torch.equal(fake.masks[0], source) and torch.equal(fake.masks[1], base),
            "actual caller native/sham changed")
    config = AutoConfig.from_pretrained(BASE, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    native_ref = create_causal_mask(config, torch.empty((*source.shape, 1)), source,
                                    torch.arange(source.shape[1]), None,
                                    position_ids=full["position_ids"][0])
    require(isinstance(native_ref, torch.Tensor) and torch.equal(base, native_ref),
            "installed native boolean SDPA mask differs")
    checks = []
    for condition, observed in zip(CONDITIONS[g["state"]][2:], fake.masks[2:], strict=True):
        verify_mask(base, observed, prompt, g, condition)
        key = g["earlier"] if condition == "earlier-row-mask" else g["latest"]
        q = g["query"]
        rect = (3,0,slice(prompt+q[0],prompt+q[1]),slice(prompt+key[0],prompt+key[1]))
        require(int(observed[rect].sum()) == 0 and int(base[rect].sum()) ==
                (q[1]-q[0])*(key[1]-key[0]), "selected mask rectangle incomplete")
    for label, changes in (("wrong_target", {"target": 2}),
                           ("wrong_query", {"query": (g["query"][0]-1,g["query"][1])}),
                           ("wrong_key", {"key": (g["latest"][0],g["latest"][1]-1)}),
                           ("wrong_prompt", {"prompt": prompt-1})):
        bad_prompt=changes.pop("prompt",prompt)
        try: mask_for(full,bad_prompt,g,"latest-row-mask",source_hash=source_hash,**changes)
        except ValueError: checks.append(label)
        else: raise AssertionError(f"actual caller accepted {label}")
    changed={**full,"attention_mask":source.clone()};changed["attention_mask"][0,0]=1-changed["attention_mask"][0,0]
    try: mask_for(changed,prompt,g,"latest-row-mask",source_hash=source_hash)
    except ValueError: checks.append("changed_source_mask")
    else: raise AssertionError("actual caller accepted changed source mask")
    for label,at in (("companion",(0,0,prompt+g["query"][0],prompt)),
                     ("prior_history_query",(3,0,prompt+g["query"][0]-1,prompt)),
                     ("current_other_key",(3,0,prompt+g["query"][0],prompt+g["query"][0]))):
        bad=fake.masks[2].clone();bad[at]=False
        try: verify_mask(base,bad,prompt,g,"latest-row-mask")
        except ValueError: checks.append(label)
        else: raise AssertionError(f"mask verifier accepted {label}")
    if g["earlier"] is not None:
        key=g["earlier"];q=g["query"]
        require(int(fake.masks[2][3,0,prompt+q[0]:prompt+q[1],prompt+key[0]:prompt+key[1]].sum())==
                (q[1]-q[0])*(key[1]-key[0]), "latest mask altered earlier row")
        require(int(fake.masks[3][3,0,prompt+q[0]:prompt+q[1],prompt+g["latest"][0]:prompt+g["latest"][1]].sum())==
                (q[1]-q[0])*(g["latest"][1]-g["latest"][0]), "earlier mask altered latest row")
        checks.append("two_rows_distinct")
    require(tensor_hash(source)==source_hash,"CPU fixture mutated source mask")
    return checks


def preflight():
    admission, registry = contract()
    require(not PREFLIGHT.exists() and not OUT.exists(), "remaining preflight/output already exists")
    q=load_qwen_components_from_options(QwenLoadOptions(base_model=str(BASE),dtype="fp32",
        attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preparation loaded model")
    shapes={};inputs={};sources={};checks={}
    for state,g0 in GEOMETRY.items():
        g={**g0,"state":state};image=registry["images"][g["image"]]
        batch,raw,_trace,receipt,planning=prepare_source(q,image,g,torch.device("cpu"))
        full,prompt=source_inputs(first.ConfigOnlyRope(),batch,raw,g,int(q.tokenizer.pad_token_id))
        shapes[state]={"batch":4,"target":3,"width":full["input_ids"].shape[1],"prompt_width":prompt,
                       "query_physical":[prompt+x for x in g["query"]],
                       "latest_physical":[prompt+x for x in g["latest"]],
                       "earlier_physical":None if g["earlier"] is None else [prompt+x for x in g["earlier"]],
                       "pixel_elements":batch.inputs["pixel_values"].numel(),
                       "image_grids":[list(x) for x in batch.image_grids],
                       "raw_lengths":g["lengths"],"active_trace_rows":[i for i,r in enumerate(raw) if len(r["token_ids"])>g["query"][1]]}
        inputs[state]={k:tensor_hash(full[k]) for k in ("input_ids","attention_mask","position_ids","cache_position")}
        sources[state]={"bindings":image["source_bindings"],"effective_identity":receipt["identity"],
                        "input_identity":receipt["input_identity"],"planning":planning}
        checks[state]=cpu_fixture(full,prompt,g)
    first_width=json.loads(first.PREFLIGHT.read_text())["shape"]["width"]
    forecast={state:admission["prior_panel_charge_gpu_seconds"]*2*(len(CONDITIONS[state])/3)*
              (shapes[state]["width"]/first_width)**2*(shapes[state]["pixel_elements"]/24403968)
              for state in GEOMETRY}
    require(all(abs(forecast[s]-admission["planning_forecast_seconds"][s])<1e-6 for s in GEOMETRY),
            "actual source-shape forecast differs from lead estimate")
    require(admission["prior_panel_charge_gpu_seconds"]+sum(forecast.values())<900 and
            admission["prior_sequence_gpu_hours"]+admission["remaining_panel_gpu_seconds"]/3600<8,
            "panel/sequence cost estimate exceeds caps")
    vocab=max(AutoConfig.from_pretrained(BASE,local_files_only=True).text_config.vocab_size,152670)
    hidden=AutoConfig.from_pretrained(BASE,local_files_only=True).text_config.hidden_size
    tensor_bytes=sum(len(CONDITIONS[s])*(4*vocab+28*(GEOMETRY[s]["query"][0]+3)*hidden)*4 for s in GEOMETRY)
    artifact_forecast=tensor_bytes*2+8_000_000
    require(artifact_forecast<256*1024*1024,"whole-panel artifact forecast exceeds 256MiB")
    direct=[Path(__file__),Path(first.__file__),Path(inspect.getfile(_source)),
            Path(inspect.getfile(_prefix_tokens)),Path(inspect.getfile(token_hash)),
            Path(inspect.getfile(_trace_compare)),
            first.REPO/'probes/recurrence_dynamics/recurrence_first_arrivals/prepare.py',
            Path(inspect.getfile(load_model)),Path(inspect.getfile(exact_history_inputs)),
            Path(inspect.getfile(input_identity)),Path(inspect.getfile(binding)),
            Path(inspect.getfile(preserve_source)),Path(inspect.getfile(load_qwen_components_from_options)),
            Path(inspect.getfile(modeling_qwen3_vl)),Path(inspect.getfile(create_causal_mask)),
            Path(inspect.getfile(sdpa_attention))]
    captures=[]
    for path in dict.fromkeys(direct):
        rel=path.relative_to(first.REPO) if path.is_relative_to(first.REPO) else Path("transformers")/path.name
        kept=preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":binding(path),"capture":binding(kept)})
    packet={"schema":"recurrence_native_history_read.remaining_preflight.v1","status":"cpu_qualified_before_gpu",
            "admission":binding(ADMISSION),"protocol":binding(first.PROTOCOL),"producer":binding(Path(__file__)),
            "first_case_producer":binding(Path(first.__file__)),"shapes":shapes,"inputs":inputs,
            "sources":sources,"cpu_actual_caller_checks":checks,"forecast_seconds":forecast,
            "forecast_total_seconds":sum(forecast.values()),"prior_charge_seconds":admission["prior_panel_charge_gpu_seconds"],
            "panel_cap_seconds":900,"artifact_forecast_bytes":artifact_forecast,
            "artifact_cap_bytes":256*1024*1024,"direct_source_captures":captures,
            "commands":{"gpu":["python","-B","-m","probes.recurrence_dynamics.recurrence_native_history_read.remaining","run"],
                        "readback":["python","-B","-m","probes.recurrence_dynamics.recurrence_native_history_read.remaining","readback"]}}
    first.write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"shapes":shapes,"forecast_seconds":forecast,
                      "artifact_forecast_bytes":artifact_forecast,"checks":checks,"captures":len(captures)}))


def checked_preflight():
    admission,registry=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["admission"]==binding(ADMISSION) and
            p["protocol"]==binding(first.PROTOCOL) and p["producer"]==binding(Path(__file__)) and
            p["first_case_producer"]==binding(Path(first.__file__)),"preflight/producer changed")
    for item in p["direct_source_captures"]:
        require(binding(item["maintained"]["path"])==item["maintained"] and
                binding(item["capture"]["path"])==item["capture"],"captured import changed")
    return admission,registry,p


def run():
    admission,registry,p=checked_preflight()
    require(not OUT.exists(),"remaining output exists; no retry")
    OUT.mkdir(parents=True)
    started=time.monotonic();counts={"model_forwards":0,"vision_forwards":0,"free_tokens":0}
    receipt={"schema":"recurrence_native_history_read.remaining_attempt.v1","status":"running",
             "pid":os.getpid(),"begun_unix":time.time(),"admission":binding(ADMISSION),
             "preflight":binding(PREFLIGHT),"producer":binding(Path(__file__)),
             "conditions":[[s,c] for s,cs in CONDITIONS.items() for c in cs],"counts":counts,"cells":[]}
    first.write_new(OUT/"launch.json",receipt)
    device=torch.device("cuda:0");handles=[]
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        q,identity=load_model("untied",device);model=q.model.eval()
        layers=[m for m in model.modules() if isinstance(m,modeling_qwen3_vl.Qwen3VLTextDecoderLayer)]
        attns=[m for m in model.modules() if isinstance(m,modeling_qwen3_vl.Qwen3VLTextAttention)]
        require(len(layers)==len(attns)==28 and [m.self_attn for m in layers]==attns,
                "actual 28-layer SDPA route changed")
        active={"full":None,"prompt":None,"query":None,"expected_mask":None,
                "seen":[],"history":[],"companions":[],"top_input":None}
        def top_hook(_module,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=7 and time.monotonic()-started < admission["remaining_panel_gpu_seconds"],
                    "model-forward/panel time cap")
            full=active["full"]
            require(torch.equal(kwargs["input_ids"],full["input_ids"]) and
                    torch.equal(kwargs["position_ids"],full["position_ids"]),"actual tokens/positions changed")
            active["top_input"]={k:tensor_hash(kwargs[k]) for k in
                                 ("input_ids","attention_mask","position_ids","cache_position")}
        def vision_hook(_module,_args):
            counts["vision_forwards"]+=1
            require(counts["vision_forwards"]<=7,"vision-forward cap")
        handles.extend((model.register_forward_pre_hook(top_hook,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(vision_hook)))
        for i,attn in enumerate(attns):
            def attn_hook(_module,_args,kwargs,layer=i):
                m=kwargs.get("attention_mask")
                require(isinstance(m,torch.Tensor) and m.ndim==4 and
                        torch.equal(m,active["expected_mask"]),f"layer {layer} consumed wrong mask")
                active["seen"].append(layer)
            handles.append(attn.register_forward_pre_hook(attn_hook,with_kwargs=True))
        for i,layer in enumerate(layers):
            def layer_hook(_module,_args,output,index=i):
                states=output[0] if isinstance(output,tuple) else output
                require(isinstance(states,torch.Tensor) and states.ndim==3,"historical layer output unavailable")
                prompt=active["prompt"];prior=active["query"][0]
                active["history"].append(states[3,prompt:prompt+prior].detach().cpu().float())
                active["companions"].append(states[[0,1,2],-1].detach().cpu().float())
            handles.append(layer.register_forward_hook(layer_hook))
        receipt["effective_identity"]=identity
        for state,g0 in GEOMETRY.items():
            g={**g0,"state":state};image=registry["images"][g["image"]]
            batch,raw,trace,source_receipt,planning=prepare_source(q,image,g,device)
            full,prompt=source_inputs(model,batch,raw,g,int(q.tokenizer.pad_token_id))
            expected=p["sources"][state]["effective_identity"]
            require({k:v for k,v in identity.items() if k!="loader_source"}==
                    {k:v for k,v in expected.items() if k!="loader_source"} and
                    all(identity["loader_source"][k]==expected["loader_source"][k]
                        for k in ("sha256","size_bytes")),"effective source/model identity changed")
            require(source_receipt["input_identity"]==p["sources"][state]["input_identity"] and
                    planning==p["sources"][state]["planning"] and
                    prompt==p["shapes"][state]["prompt_width"] and
                    all(tensor_hash(full[k])==p["inputs"][state][k] for k in
                        ("input_ids","attention_mask","position_ids","cache_position")),
                    f"{state} source inputs differ from CPU preflight")
            first.write_new(OUT/f"inputs-{state}.json",{
                "input_ids":full["input_ids"].cpu().tolist(),
                "attention_mask":full["attention_mask"].cpu().tolist(),
                "position_ids":full["position_ids"].cpu().tolist(),
                "cache_position":full["cache_position"].cpu().tolist()})
            source_hash=tensor_hash(full["attention_mask"])
            native=None
            with torch.inference_mode():
                for condition in CONDITIONS[state]:
                    remaining=admission["remaining_panel_gpu_seconds"]-(time.monotonic()-started)
                    per_call=p["forecast_seconds"][state]/len(CONDITIONS[state])
                    require(remaining>per_call,"insufficient panel budget for next frozen call")
                    active.update(full=full,prompt=prompt,query=g["query"],seen=[],history=[],
                                  companions=[],top_input=None,expected_mask=None)
                    output=forward_cell(model,full,prompt,g,condition,active,source_hash)
                    torch.cuda.synchronize(device)
                    require(active["seen"]==list(range(28)) and len(active["history"])==
                            len(active["companions"])==28 and active["top_input"] is not None,
                            "actual all-layer consumer incomplete")
                    logits=output.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape[0]==4 and torch.isfinite(logits).all().item(),
                            "invalid full-vocabulary endpoint")
                    payload={"state":state,"condition":condition,"logits":logits,
                             "all_prior_history_by_layer":torch.stack(active["history"]),
                             "companion_last_by_layer":torch.stack(active["companions"])}
                    torch.save(payload,OUT/f"{state}-{condition}.pt")
                    key=g["earlier"] if condition=="earlier-row-mask" else g["latest"]
                    rect=(3,0,slice(prompt+g["query"][0],prompt+g["query"][1]),
                          slice(prompt+key[0],prompt+key[1]))
                    _,base=mask_for(full,prompt,g,condition,source_hash=source_hash)
                    expected_mask=base if condition=="native" else active["expected_mask"]
                    complement=expected_mask.clone();complement[rect]=False
                    cell={"state":state,"condition":condition,"raw":binding(OUT/f"{state}-{condition}.pt"),
                          "actual_input_hashes":active["top_input"],
                          "actual_mask_layers":active["seen"],
                          "query_physical":[prompt+x for x in g["query"]],
                          "key_physical":[prompt+x for x in key],
                          "selected_true_count":int(expected_mask[rect].sum()),
                          "complement_sha256":tensor_hash(complement),
                          "elapsed_internal_gpu_seconds":time.monotonic()-started}
                    if condition=="native":
                        active_rows=[i for i,r in enumerate(raw) if len(r["token_ids"])>g["query"][1]]
                        cell["source_trace_parity"]=[_trace_compare(
                            logits=logits[i],trace=trace,batch_index=i,
                            absolute_offset=g["query"][1],token_id=raw[i]["token_ids"][g["query"][1]],
                            role="native_unforced_x1") for i in active_rows]
                        require(all(x["passed"] for x in cell["source_trace_parity"]),
                                f"{state} source parity failed before mask cells")
                        native=payload
                    else:
                        require(native is not None,"missing native reference")
                        row_error=float((payload["all_prior_history_by_layer"]-
                                         native["all_prior_history_by_layer"]).abs().max())
                        companion_hidden_error=float((payload["companion_last_by_layer"]-
                                                      native["companion_last_by_layer"]).abs().max())
                        cell["all_prior_history_max_error"]=row_error
                        cell["companion_last_states_max_error"]=companion_hidden_error
                        require(max(row_error,companion_hidden_error)<=TOL,
                                f"{state} historical or companion layer states changed")
                        if condition=="identity-mask-sham":
                            cell["all_batch_vector_max_error"]=float((logits-native["logits"]).abs().max())
                            require(cell["all_batch_vector_max_error"]<=TOL,
                                    f"{state} sham full vectors differ")
                        else:
                            cell["companion_vector_max_errors"]=[
                                float((logits[i]-native["logits"][i]).abs().max()) for i in (0,1,2)]
                            require(max(cell["companion_vector_max_errors"])<=TOL,
                                    f"{state} treatment companion vector changed")
                            expected_count=(g["query"][1]-g["query"][0])*(key[1]-key[0])
                            require(cell["selected_true_count"]==0 and
                                    native_cell["selected_true_count"]==expected_count and
                                    cell["complement_sha256"]==native_cell["complement_sha256"],
                                    f"{state} selected/complement mask scope changed")
                    if condition=="native": native_cell=cell
                    receipt["cells"].append(cell)
                    receipt["counts"]=dict(counts)
                    receipt["allocated_gpu_seconds"]=time.monotonic()-started
                    first.write_new(OUT/f"checkpoint-{len(receipt['cells'])}.json",receipt)
        require(counts=={"model_forwards":7,"vision_forwards":7,"free_tokens":0} and
                len(receipt["cells"])==7,"exact seven-call queue incomplete")
        receipt["status"]="candidate_complete"
    except BaseException as exc:
        receipt["status"]="technical_invalid"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    finally:
        for handle in handles: handle.remove()
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        receipt["allocated_gpu_seconds"]=time.monotonic()-started
        receipt["counts"]=dict(counts)
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["terminal_pid"]=os.getpid()
        receipt["artifact_bytes_before_receipt"]=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
        first.write_new(OUT/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"counts":counts,"allocated_gpu_seconds":receipt["allocated_gpu_seconds"],
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_complete","remaining queue failed; no retry")


def readback():
    admission,registry,p=checked_preflight()
    receipt=json.loads((OUT/"receipt.json").read_text())
    require(receipt["status"]=="candidate_complete" and
            receipt["counts"]=={"model_forwards":7,"vision_forwards":7,"free_tokens":0} and
            len(receipt["cells"])==7 and receipt["admission"]==binding(ADMISSION) and
            not (OUT/"readback.json").exists(),"remaining attempt incomplete/changed")
    summaries={}
    for state,g0 in GEOMETRY.items():
        g={**g0,"state":state};image=registry["images"][g["image"]]
        saved=json.loads((OUT/f"inputs-{state}.json").read_text())
        full={k:torch.tensor(saved[k]) for k in ("input_ids","attention_mask","position_ids","cache_position")}
        require(all(tensor_hash(full[k])==p["inputs"][state][k] for k in full),
                f"{state} cold input/position changed")
        prompt=p["shapes"][state]["prompt_width"]
        cells=[c for c in receipt["cells"] if c["state"]==state]
        require([c["condition"] for c in cells]==list(CONDITIONS[state]),f"{state} cell order changed")
        vectors={}
        complement_hashes=[]
        for cell in cells:
            c=cell["condition"]
            require(binding(cell["raw"]["path"])==cell["raw"] and
                    cell["actual_mask_layers"]==list(range(28)),"raw/layer evidence changed")
            mask,base=mask_for(full,prompt,g,c,source_hash=p["inputs"][state]["attention_mask"])
            viewed=base if c=="native" else mask
            key=g["earlier"] if c=="earlier-row-mask" else g["latest"]
            rect=(3,0,slice(prompt+g["query"][0],prompt+g["query"][1]),
                  slice(prompt+key[0],prompt+key[1]))
            complement=viewed.clone();complement[rect]=False
            require(cell["actual_input_hashes"]=={
                "input_ids":p["inputs"][state]["input_ids"],
                "attention_mask":tensor_hash(mask),
                "position_ids":p["inputs"][state]["position_ids"],
                "cache_position":p["inputs"][state]["cache_position"]} and
                cell["selected_true_count"]==int(viewed[rect].sum()) and
                cell["complement_sha256"]==tensor_hash(complement),
                f"{state}:{c} actual input/mask failed cold reconstruction")
            complement_hashes.append(cell["complement_sha256"])
            vectors[c]=torch.load(cell["raw"]["path"],map_location="cpu",weights_only=True)
        require(len(set(complement_hashes))==1,f"{state} mask complements differ")
        native=vectors["native"]
        require(float((native["logits"]-vectors["identity-mask-sham"]["logits"]).abs().max())<=TOL,
                f"{state} cold sham vectors differ")
        for c in CONDITIONS[state][1:]:
            x=vectors[c]
            require(float((x["all_prior_history_by_layer"]-native["all_prior_history_by_layer"]).abs().max())<=TOL and
                    float((x["companion_last_by_layer"]-native["companion_last_by_layer"]).abs().max())<=TOL,
                    f"{state}:{c} prior/companion layer states changed")
            if c!="identity-mask-sham":
                require(all(float((x["logits"][i]-native["logits"][i]).abs().max())<=TOL for i in (0,1,2)),
                        f"{state}:{c} companion full vector changed")
        raw=json.loads(Path(image["source_bindings"]["raw"]["path"]).read_text())["rows"]
        trace=json.loads(Path(image["source_bindings"]["trace"]["path"]).read_text())
        active=[i for i,r in enumerate(raw) if len(r["token_ids"])>g["query"][1]]
        parity=[_trace_compare(logits=native["logits"][i],trace=trace,batch_index=i,
                               absolute_offset=g["query"][1],token_id=raw[i]["token_ids"][g["query"][1]],
                               role="cold_native_x1") for i in active]
        require(all(x["passed"] for x in parity),f"{state} cold source trace parity failed")
        def stats(logits):
            v=logits[3].double();probs=torch.softmax(v,-1);logp=torch.log_softmax(v,-1);top=torch.topk(v,2)
            ids=[g["old"],g["new"]]+([] if g["old0"] is None else [g["old0"]])
            return {"global_top2_ids":top.indices.tolist(),"global_top2_logits":top.values.tolist(),
                    "global_gap":float(top.values[0]-top.values[1]),
                    "tokens":{str(t):{"logit":float(v[t]),"prob":float(probs[t]),
                                      "logprob":float(logp[t]),"rank":int((v>v[t]).sum())+1} for t in ids}}
        observations={c:stats(vectors[c]["logits"]) for c in CONDITIONS[state]}
        margins={c:float(vectors[c]["logits"][3,g["new"]].double()-
                         vectors[c]["logits"][3,g["old"]].double()) for c in CONDITIONS[state]}
        native_p=torch.softmax(native["logits"][3].double(),-1)
        treatments={}
        for c in CONDITIONS[state][2:]:
            q=torch.softmax(vectors[c]["logits"][3].double(),-1)
            treatments[c]={"primary_delta_mask_minus_native":margins[c]-margins["native"],
                           "full_vocabulary_tv":float(.5*(native_p-q).abs().sum())}
        secondary=None
        if g["old0"] is not None:
            secondary={c:float(vectors[c]["logits"][3,g["new"]].double()-
                               vectors[c]["logits"][3,g["old0"]].double()) for c in CONDITIONS[state]}
            for c in treatments:
                treatments[c]["secondary_B_minus_A0_delta"]=secondary[c]-secondary["native"]
        summaries[state]={"source_parity":parity,"observations":observations,
                          "primary_B_minus_A1_or_cow_B_minus_A":margins,
                          "secondary_bowl_B_minus_A0":secondary,"treatments":treatments,
                          "complement_sha256":complement_hashes[0]}
    first_readback=json.loads(Path(json.loads(Path(admission["first_case_acceptance"]["path"]).read_text())["readback"]["path"]).read_text())
    accepted_delta=first_readback["primary_B_minus_A1"]["delta_mask_minus_native"]
    latest_delta=summaries["bowl-row2"]["treatments"]["latest-row-mask"]["primary_delta_mask_minus_native"]
    summary={"schema":"recurrence_native_history_read.remaining_readback.v1","status":"cold_readback_passed",
             "receipt":binding(OUT/"receipt.json"),"admission":binding(ADMISSION),
             "first_case_readback":binding(Path(json.loads(Path(admission["first_case_acceptance"]["path"]).read_text())["readback"]["path"])),
             "states":summaries,"bowl_latest_row2_minus_accepted_row1_delta":latest_delta-accepted_delta,
             "counts":receipt["counts"],"internal_gpu_seconds":receipt["allocated_gpu_seconds"],
             "prior_panel_charge_seconds":admission["prior_panel_charge_gpu_seconds"]}
    first.write_new(OUT/"readback.json",summary)
    print(json.dumps({"status":summary["status"],"bowl_latest_delta":latest_delta,
                      "bowl_row2_minus_row1_delta":summary["bowl_latest_row2_minus_accepted_row1_delta"],
                      "cow_delta":summaries["cow-row1"]["treatments"]["latest-row-mask"]["primary_delta_mask_minus_native"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("preflight","run","readback"))
    {"preflight":preflight,"run":run,"readback":readback}[parser.parse_args().action]()
