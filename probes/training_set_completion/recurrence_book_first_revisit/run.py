"""Two fixed book states: free current row with immediately prior row unreadable."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.coordinate_continuity.runtime import _source, tensor_hash, token_hash
from probes.training_set_completion.native_row_choice.runtime import _trace_compare
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_native_history_read.run import ConfigOnlyRope, native_4d
from probes.training_set_completion.recurrence_free_header_routing.run import parse_row
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity
from src.qwen.native import exact_history_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options

REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-book-first-revisit"
ADMISSION = UNIT / "lead-admission-v1.json"
PROTOCOL = UNIT / "native-read-protocol.md"
PREFLIGHT = UNIT / "supporting/native-read-preflight-v1.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-book-first-revisit/native-read-v1")
SHAS = {ADMISSION: "08fc23ff2693d3597263a72fb3f8b7c57348ffed7b2f731e8ee9a328574284ea",
        PROTOCOL: "e74467fa451e157c5a8f118eb1069c0fa3d21251a83681df7292674a71a616aa"}
STATES = ("row4_first_B_revisit", "row3_new_B")
ARMS = ("native", "identity-mask-sham", "latest-row-reading-mask")
TARGET, PROMPT, TOL = 1, 1362, 2e-4
KEYS = ("input_ids", "attention_mask", "position_ids", "cache_position")


def bind(path):
    return literal_binding(Path(path))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush(); os.fsync(file.fileno())
    return bind(path)


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-finite-two-state-native-reading-package" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["raw_output_root"] == str(OUT) and
            [s["state_id"] for s in a["ordered_cases"]] == list(STATES) and
            a["arms"] == list(ARMS) and
            a["max_model_forwards"] == a["max_vision_forwards"] == 66 and
            a["max_logical_emitted_tokens"] == 68 and a["max_width"] == 1413,
            "admitted finite queue changed")
    for name in ("protocol", "cpu_protocol", "cpu_acceptance", "cpu_candidate", "cpu_bindings", "source_panel"):
        b = a[name]; require(bind(b["path"]) == b, f"admission dependency changed: {name}")
    for b in a["source_bindings"].values():
        require(bind(b["path"]) == b, "source binding changed")
    for b in a["historical_loader_path_crosswalk"]:
        maintained = bind(b["maintained_path"])
        require(maintained["sha256"] == b["sha256"] == b["historical"]["sha256"] and
                maintained["size_bytes"] == b["size_bytes"] == b["historical"]["size_bytes"],
                "maintained loader crosswalk failed")
    return a


def source(q, a, device):
    b = a["source_bindings"]
    rows = json.loads(Path(b["raw"]["path"]).read_text())["rows"]
    boundary = {"group": "new-08", "batch_index": TARGET, "image_id": 151704,
                "raw_path": b["raw"]["path"], "trace_path": b["trace"]["path"],
                "receipt_path": b["runtime_receipt"]["path"],
                "native_tokens": rows[TARGET]["token_ids"],
                "native_token_hash": token_hash(rows[TARGET]["token_ids"])}
    panel = json.loads(Path(a["source_panel"]["path"]).read_text())
    batch, raw, trace, group, planning = _source(boundary, "untied", panel, q, device)
    receipt = json.loads(Path(b["runtime_receipt"]["path"]).read_text())
    expected = a["source_full_batch"]
    require(len(raw) == len(group["cases"]) == 4 and
            list(batch.request_ids) == expected["request_ids"] and
            [len(x["token_ids"]) for x in raw] == expected["raw_lengths"] and
            list(map(len,batch.prompt_token_ids)) == expected["prompt_lengths"] and
            int(batch.inputs["pixel_values"].numel()) == expected["pixel_elements"] and
            input_identity(batch) == receipt["input_identity"] and
            planning["replanned_image_plan"] is True and
            all(raw[TARGET]["token_ids"][s["prefix_end"]:s["prefix_end"]+9] ==
                s["native_expected_ids"] for s in a["ordered_cases"]),
            "original full batch/source rows changed")
    return batch,raw,trace,receipt,planning


def inputs(model,batch,raw,pad,state,emitted,*,previous=None,target=TARGET,prefix=None):
    p = state["prefix_end"] if prefix is None else prefix
    t = len(emitted)
    require(target == TARGET and p == state["prefix_end"] and
            state["latest_raw_keys"] == [p-9,p] and 0 <= t < 16 and
            (t == 0 or emitted[-1] == previous), "wrong target/prefix/own greedy token")
    tails = _prefix_tokens(raw,p+t,pad)
    tails[TARGET] = raw[TARGET]["token_ids"][:p] + list(emitted)
    histories = [list(prompt)+tail for prompt,tail in zip(batch.prompt_token_ids,tails,strict=True)]
    full = exact_history_inputs(model,batch.inputs,histories,pad_token_id=pad,logits_to_keep=1)
    w = int(full["input_ids"].shape[1])
    full["cache_position"] = torch.arange(w,device=full["input_ids"].device)
    require(w == PROMPT+p+t and full["attention_mask"].shape == (4,w) and
            full["position_ids"].shape == (3,4,w) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            tails[TARGET][p:] == list(emitted) and
            all(tails[i] == raw[i]["token_ids"][:p+t] for i in (0,2,3)) and
            all(full["attention_mask"][i,:PROMPT].sum().item() == len(batch.prompt_token_ids[i]) for i in range(4)) and
            full["attention_mask"][:,PROMPT:].all().item(),
            "full batch source/positions/prefix changed")
    return full


def masks(full,state,arm,*,target=TARGET,query=None,key=None,source_hash=None):
    p = state["prefix_end"]; t = int(full["input_ids"].shape[1])-PROMPT-p
    expected_query = (PROMPT+p,PROMPT+p+t)
    expected_key = (PROMPT+p-9,PROMPT+p)
    require(arm in ARMS and target == TARGET and query in (None,expected_query) and
            key in (None,expected_key) and t >= 0 and
            (source_hash is None or tensor_hash(full["attention_mask"]) == source_hash),
            "wrong target/query/key/source mask")
    native = native_4d(full["attention_mask"])
    selected = torch.zeros_like(native)
    selected[TARGET,0,expected_query[0]:expected_query[1],expected_key[0]:expected_key[1]] = True
    require(int(selected.sum()) == 9*t and (t == 0 or bool(native[selected].all())),
            "selected rectangle is not native-readable")
    actual = native.clone()
    if arm == "latest-row-reading-mask": actual[selected] = False
    verify_rectangle(native,actual,selected,arm,t)
    return full["attention_mask"] if arm == "native" else actual,native,selected


def verify_rectangle(native,actual,selected,arm,t):
    require(native.shape == actual.shape == selected.shape and
            int(selected.sum()) == 9*t and
            torch.equal(actual[~selected],native[~selected]) and
            torch.equal(actual[selected],torch.zeros_like(native[selected]) if arm == ARMS[2] else native[selected]),
            "same-rectangle selected/complement mask changed")


def caller(model,full,state,arm,observer):
    source_hash=tensor_hash(full["attention_mask"])
    mask,native,selected=masks(full,state,arm,source_hash=source_hash)
    payload={**full,"attention_mask":mask}
    observer.update(expected=mask if arm != "native" else native,
                    native=native,selected=selected,source_hash=source_hash,
                    actual=None,layers=[],history=[],companions=[])
    return model(**payload)


def verify_input(model,batch,raw,pad,state,arm,emitted,full,observed,logits,chosen,*,previous=None):
    expected=inputs(model,batch,raw,pad,state,emitted,previous=previous)
    input_hashes={k:tensor_hash(expected[k]) for k in KEYS}
    require(all(torch.equal(full[k],expected[k]) for k in KEYS) and
            observed["actual"] == {**input_hashes,"attention_mask":tensor_hash(
                masks(expected,state,arm)[0])} and
            observed["layers"] == list(range(28)) and
            chosen == int(torch.argmax(logits[TARGET]).item()),
            "actual input/greedy/28-layer consumer differs")
    mask,native,selected=masks(expected,state,arm)
    require(torch.equal(observed["expected"],native if arm == "native" else mask) and
            torch.equal(observed["selected"],selected), "actual mask receipt changed")
    return input_hashes


def cpu_checks(batch,raw,pad,a,special):
    config=AutoConfig.from_pretrained(BASE,local_files_only=True).text_config
    config._attn_implementation="sdpa"
    model=ConfigOnlyRope(); checks=[]
    require(parse_row([151646,2190,151647,151648,151670,151671,151672,151673,151649],special)["stop"]=="complete" and
            parse_row([151645],special)["stop"]=="eos" and
            parse_row([151649],special)["stop"]=="early_row_terminator" and
            parse_row([151646,2190,151647,151648,152670],special)["stop"]=="malformed_coordinate" and
            parse_row([151646,2190,151647,151648,152669],special)["stop"] is None and
            parse_row([151646]+[2190]*15,special)["stop"]=="cap" and
            parse_row([151646,2190,151647,151648,151670,151671,151672,151673,151670],special)["stop"]=="malformed_terminator",
            "free row parser boundaries changed")
    checks.append("parser_boundaries")
    for state in a["ordered_cases"]:
        for t in (0,1,15):
            emitted=list(state["native_expected_ids"][:t]) if t<=9 else [151646]+[2190]*(t-1)
            full=inputs(model,batch,raw,pad,state,emitted,previous=emitted[-1] if t else None)
            native=native_4d(full["attention_mask"])
            reference=create_causal_mask(config,torch.empty((*full["attention_mask"].shape,1)),
                full["attention_mask"],full["cache_position"],None,position_ids=full["position_ids"][0])
            require(torch.equal(native,reference),"installed SDPA mask convention differs")
            for arm in ARMS:
                class Fake:
                    def __call__(self,**kwargs):
                        seen["actual"]={k:tensor_hash(kwargs[k]) for k in KEYS}
                        out=torch.zeros((4,152670));out[TARGET,151646]=1
                        return SimpleNamespace(logits=out[:,None,:])
                seen={}; logits=caller(Fake(),full,state,arm,seen).logits[:,-1]
                seen["layers"]=list(range(28));
                verify_input(model,batch,raw,pad,state,arm,emitted,full,seen,logits,151646,
                             previous=emitted[-1] if t else None)
                checks.append(f"actual_caller_{state['state_id']}_{arm}_{t}")
                if t==0 and arm==ARMS[2]:
                    require(torch.equal(masks(full,state,ARMS[0])[1],masks(full,state,ARMS[2])[0]),
                            "false t0 native reuse")
            for label,kw in (("target",{"target":2}),("query",{"query":(PROMPT+state["prefix_end"]-1,PROMPT+state["prefix_end"]+t)}),
                             ("key",{"key":(PROMPT+state["prefix_end"]-8,PROMPT+state["prefix_end"])})):
                try: masks(full,state,ARMS[2],**kw)
                except ValueError: checks.append(f"reject_{label}_{t}")
                else: raise AssertionError(label)
            expected,native,selected=masks(full,state,ARMS[2]); bad=expected.clone()
            bad[0,0,-1,0]=~bad[0,0,-1,0]
            try: verify_rectangle(native,bad,selected,ARMS[2],t)
            except ValueError: checks.append(f"reject_complement_{t}")
            else: raise AssertionError("complement")
            badfull={**full,"input_ids":full["input_ids"].clone()};badfull["input_ids"][0,-1]+=1
            try: verify_input(model,batch,raw,pad,state,ARMS[0],emitted,badfull,seen,logits,151646,
                              previous=emitted[-1] if t else None)
            except ValueError: checks.append(f"reject_companion_{t}")
            else: raise AssertionError("companion")
            badfull={**full,"position_ids":full["position_ids"].clone()};badfull["position_ids"][0,TARGET,-1]+=1
            try: verify_input(model,batch,raw,pad,state,ARMS[0],emitted,badfull,seen,logits,151646,
                              previous=emitted[-1] if t else None)
            except ValueError: checks.append(f"reject_position_{t}")
            else: raise AssertionError("position")
            if t:
                try: inputs(model,batch,raw,pad,state,emitted,previous=-1)
                except ValueError: checks.append("reject_own_token")
                else: raise AssertionError("own token")
            try: inputs(model,batch,raw,pad,state,emitted,prefix=state["prefix_end"]+1)
            except ValueError: checks.append("reject_prefix")
            else: raise AssertionError("prefix")
    return checks


def preflight():
    a=contract();require(not PREFLIGHT.exists() and not OUT.exists(),"preflight/run already exists")
    q=load_qwen_components_from_options(QwenLoadOptions(base_model=str(BASE),dtype="fp32",
        attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded model")
    batch,raw,trace,receipt,planning=source(q,a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id); special=frozenset(q.tokenizer.all_special_ids)
    checks=cpu_checks(batch,raw,pad,a,special)
    maxwidth=max(PROMPT+s["prefix_end"]+15 for s in a["ordered_cases"])
    require(maxwidth==a["max_width"] and pad==a["source_full_batch"]["pad_id"] and
            len(checks)>=40 and a["planning_outer_gpu_seconds"]<1000 and
            a["artifact_planning_envelope_bytes"]>=1024**3,
            "shape/cost/source forecast changed")
    sources=[Path(__file__),Path(inspect.getfile(_source)),Path(inspect.getfile(_prefix_tokens)),
        Path(inspect.getfile(_trace_compare)),Path(inspect.getfile(parse_row)),
        Path(inspect.getfile(ConfigOnlyRope)),Path(inspect.getfile(native_4d)),
        Path(inspect.getfile(load_model)),Path(inspect.getfile(exact_history_inputs)),
        Path(inspect.getfile(input_identity)),Path(inspect.getfile(preserve_source)),
        Path(inspect.getfile(bind)),Path(inspect.getfile(create_causal_mask)),
        Path(inspect.getfile(modeling_qwen3_vl)),REPO/"src/qwen/untied_embeddings.py",
        REPO/"src/qwen/runtime_loading.py",REPO/"src/qwen/native.py",
        REPO/"src/inference/inputs.py",REPO/"src/inference/bound_requests.py",
        REPO/"src/data/examples.py",REPO/"src/config/inference.py"]
    captures=[]
    for path in dict.fromkeys(sources):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_book_first_revisit.run"]
    packet={"status":"cpu_qualified_before_gpu","admission":bind(ADMISSION),"protocol":bind(PROTOCOL),
        "producer":bind(Path(__file__)),"source_identity":receipt["identity"],
        "input_identity":receipt["input_identity"],"planning":planning,
        "full_batch":a["source_full_batch"],"pad":pad,"special":sorted(special),
        "cpu_checks":checks,"max_width":maxwidth,"planning_outer_seconds":a["planning_outer_gpu_seconds"],
        "artifact_envelope_bytes":a["artifact_planning_envelope_bytes"],"captures":captures,
        "commands":{name:command+[name] for name in ("run_row4","readback_row4","run_row3","readback_row3")}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),"captures":len(captures),"max_width":maxwidth}))


def checked():
    a=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["admission"]==bind(ADMISSION) and
            p["producer"]==bind(Path(__file__)),"preflight binding changed")
    for c in p["captures"]:
        require(bind(c["maintained"]["path"])==c["maintained"] and
                bind(c["capture"]["path"])==c["capture"],"source capture changed")
    return a,p


def run_state(state_id):
    a,p=checked();index=STATES.index(state_id);state=a["ordered_cases"][index]
    root=OUT/state_id
    require(not root.exists(),"state already launched; no retry")
    if index:
        r=json.loads((OUT/STATES[0]/"receipt.json").read_text())
        c=json.loads((OUT/STATES[0]/"readback.json").read_text())
        require(r["status"]=="candidate_complete" and c["status"]=="candidate_cold_readback_passed" and
                c["receipt"]==bind(OUT/STATES[0]/"receipt.json"),"row4 cold gate failed")
    root.mkdir(parents=True)
    started=time.monotonic();device=torch.device("cuda:0");handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"logical_tokens":0,"reused_t0":0}
    receipt={"status":"running","state":state_id,"pid":os.getpid(),"started_unix":time.time(),
             "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
             "arms":[],"counts":counts}
    write_new(root/"launch.json",receipt)
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        q,identity=load_model("untied",device)
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in p["source_identity"].items() if k!="loader_source"} and
                all(identity["loader_source"][k]==p["source_identity"]["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"]==a["historical_loader_path_crosswalk"][0]["maintained_path"],
                "effective model/maintained loader differs")
        model=q.model.eval();batch,raw,trace,sr,planning=source(q,a,device)
        require(sr["input_identity"]==p["input_identity"] and planning==p["planning"],"GPU source changed")
        pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
        require(pad==p["pad"] and sorted(special)==p["special"],"tokenizer IDs changed")
        layers=list(model.model.language_model.layers)
        attentions=[x.self_attn for x in layers]
        require(len(layers)==len(attentions)==28 and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "28 actual text layers changed")
        active={}
        def top(_m,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=33,"state model cap")
            active["actual"]={k:tensor_hash(kwargs[k]) for k in KEYS}
            require(active["actual"]=={**{k:tensor_hash(active["full"][k]) for k in KEYS},
                   "attention_mask":tensor_hash(active["expected"] if active["arm"]!="native" else active["full"]["attention_mask"])},
                   "top-level actual input differs")
        def vision(_m,_args):
            counts["vision_forwards"]+=1;require(counts["vision_forwards"]<=33,"state vision cap")
        handles += [model.register_forward_pre_hook(top,with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(vision)]
        for i,x in enumerate(attentions):
            def hook(_m,_args,kwargs,layer=i):
                m=kwargs.get("attention_mask")
                require(isinstance(m,torch.Tensor) and m.ndim==4 and torch.equal(m,active["expected"]),
                        f"layer {layer} consumed wrong mask")
                active["layers"].append(layer)
            handles.append(x.register_forward_pre_hook(hook,with_kwargs=True))
        for i,x in enumerate(layers):
            def state_hook(_m,_args,output,layer=i):
                value=output[0] if isinstance(output,tuple) else output
                require(isinstance(value,torch.Tensor) and value.ndim==3,"hidden states unavailable")
                active["history"].append(value[TARGET,PROMPT:PROMPT+state["prefix_end"]].detach().cpu().float())
                active["companions"].append(value[[0,2,3],-1].detach().cpu().float())
            handles.append(x.register_forward_hook(state_hook))
        receipt["effective_identity"]=identity
        with torch.inference_mode():
            for arm in ARMS:
                emitted=[];record={"arm":arm,"steps":[],"stop":None};receipt["arms"].append(record)
                for t in range(16 if arm==ARMS[2] else 9):
                    previous=emitted[-1] if t else None
                    full=inputs(model,batch,raw,pad,state,emitted,previous=previous)
                    if arm==ARMS[2] and t==0:
                        native=receipt["arms"][0]["steps"][0]
                        require(native["chosen"]==151646 and
                                all(native["input_hashes"][k]==tensor_hash(full[k]) for k in KEYS) and
                                torch.equal(masks(full,state,ARMS[0])[1],masks(full,state,arm)[0]),
                                "native t0 reuse identity failed")
                        chosen=native["chosen"]
                        record["steps"].append({"step":0,"reused_native_t0":True,"native_entry":native,
                                                "chosen":chosen})
                        emitted.append(chosen);counts["logical_tokens"]+=1;counts["reused_t0"]+=1
                        record["emitted"]=list(emitted);record["stop"]=parse_row(emitted,special)["stop"]
                        write_new(root/f"checkpoint-{arm}-{t}.json",receipt)
                        continue
                    active.update(full=full,arm=arm)
                    out=caller(model,full,state,arm,active)
                    torch.cuda.synchronize(device)
                    require(len(active["layers"])==len(active["history"])==len(active["companions"])==28 and
                            active.get("actual") is not None,"actual consumers/states incomplete")
                    logits=out.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape==(4,152670) and torch.isfinite(logits).all().item(),"invalid logits")
                    chosen=int(torch.argmax(logits[TARGET]).item())
                    hashes=verify_input(model,batch,raw,pad,state,arm,emitted,full,active,logits,chosen,previous=previous)
                    rawpath=root/f"{arm}-step{t}.pt"
                    torch.save({"arm":arm,"step":t,"logits":logits,
                        "history":torch.stack(active["history"]),"companions":torch.stack(active["companions"])},rawpath)
                    inputpath=root/f"inputs-{arm}-step{t}.json"
                    write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                    entry={"step":t,"raw":bind(rawpath),"inputs":bind(inputpath),"input_hashes":hashes,
                           "actual_input":active["actual"],"actual_layers":active["layers"],
                           "actual_mask_hash":tensor_hash(active["expected"]),
                           "selected_cells":int(active["selected"].sum()),
                           "selected_native_hash":tensor_hash(active["native"][active["selected"]]),
                           "selected_actual_hash":tensor_hash(active["expected"][active["selected"]]),
                           "complement_native_hash":tensor_hash(active["native"][~active["selected"]]),
                           "complement_actual_hash":tensor_hash(active["expected"][~active["selected"]]),
                           "chosen":chosen,"internal_seconds":time.monotonic()-started}
                    if arm==ARMS[0]:
                        entry["source_parity"]=[_trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=state["prefix_end"]+t,
                            token_id=raw[i]["token_ids"][state["prefix_end"]+t],role="book_native")
                            for i in range(4)]
                        require(all(x["passed"] for x in entry["source_parity"]) and
                                chosen==state["native_expected_ids"][t],"native source parity failed")
                    else:
                        entry["companion_source_parity"]=[_trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=state["prefix_end"]+t,
                            token_id=raw[i]["token_ids"][state["prefix_end"]+t],role="book_companion")
                            for i in (0,2,3)]
                        require(all(x["passed"] for x in entry["companion_source_parity"]),
                                "companion source trace changed")
                        if t<9:
                            nr=receipt["arms"][0]["steps"][t]
                            old=torch.load(nr["raw"]["path"],map_location="cpu",weights_only=True)
                            cur=torch.load(rawpath,map_location="cpu",weights_only=True)
                            entry["companion_state_error"]=float((cur["companions"]-old["companions"]).abs().max())
                            entry["companion_vector_error"]=max(float((logits[i]-old["logits"][i]).abs().max()) for i in (0,2,3))
                            entry["history_state_error"]=float((cur["history"]-old["history"]).abs().max())
                            require(max(entry["companion_state_error"],entry["companion_vector_error"],
                                        entry["history_state_error"])<=TOL,"history/companion changed")
                            if arm==ARMS[1]:
                                entry["sham_full_vector_error"]=float((logits-old["logits"]).abs().max())
                                require(entry["sham_full_vector_error"]<=TOL and
                                        chosen==state["native_expected_ids"][t],"sham vector/greedy failed")
                    record["steps"].append(entry);emitted.append(chosen);counts["logical_tokens"]+=1
                    require(counts["logical_tokens"]<=34,"logical token cap")
                    parsed=parse_row(emitted,special)
                    record["emitted"]=list(emitted);record["stop"]=parsed["stop"]
                    write_new(root/f"checkpoint-{arm}-{t}.json",receipt)
                    if parsed["stop"] is not None:break
                require(record["stop"] is not None,"trajectory failed to stop by cap")
                if arm!=ARMS[2]:
                    require(record["stop"]=="complete" and record["emitted"]==state["native_expected_ids"],
                            "native/sham qualification failed")
        receipt["status"]="candidate_complete"
    except BaseException as exc:
        receipt["status"]="technical_invalid"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    finally:
        for h in handles:h.remove()
        if torch.cuda.is_available():torch.cuda.synchronize(device)
        receipt["counts"]=dict(counts)
        receipt["internal_seconds"]=time.monotonic()-started
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["artifact_bytes"]=sum(x.stat().st_size for x in root.rglob("*") if x.is_file())
        receipt["terminal_pid"]=os.getpid()
        write_new(root/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"state":state_id,"counts":counts,
                      "internal_seconds":receipt["internal_seconds"],
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_complete","technical failure; no retry")


def readback(state_id):
    a,p=checked();state=a["ordered_cases"][STATES.index(state_id)];root=OUT/state_id
    r=json.loads((root/"receipt.json").read_text())
    require(r["status"]=="candidate_complete" and r["state"]==state_id and
            [x["arm"] for x in r["arms"]]==list(ARMS) and not (root/"readback.json").exists(),
            "receipt/readback gate changed")
    q=load_qwen_components_from_options(QwenLoadOptions(base_model=str(BASE),dtype="fp32",
        attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold readback loaded model")
    batch,raw,trace,sr,planning=source(q,a,torch.device("cpu"))
    require(sr["input_identity"]==p["input_identity"] and planning==p["planning"],"cold source changed")
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    model=ConfigOnlyRope();native=[];arms=[];actual=0;logical=0
    for record in r["arms"]:
        arm=record["arm"];emitted=[];steps=[]
        for t,e in enumerate(record["steps"]):
            if e.get("reused_native_t0"):
                require(arm==ARMS[2] and t==0 and e["native_entry"]==r["arms"][0]["steps"][0] and
                        e["chosen"]==native[0]["chosen"],"cold native t0 reuse changed")
                emitted.append(e["chosen"]);logical+=1
                steps.append({"step":0,"chosen":e["chosen"],"reuse":"native_t0"})
                continue
            require(bind(e["raw"]["path"])==e["raw"] and bind(e["inputs"]["path"])==e["inputs"],
                    "raw/input binding changed")
            vals=json.loads(Path(e["inputs"]["path"]).read_text())
            full={k:torch.tensor(v,dtype=torch.long) for k,v in vals.items()}
            payload=torch.load(e["raw"]["path"],map_location="cpu",weights_only=True)
            logits=payload["logits"]
            require(payload["arm"]==arm and payload["step"]==t and
                    logits.shape==(4,152670) and torch.isfinite(logits).all().item() and
                    payload["history"].shape[0]==payload["companions"].shape[0]==28,
                    "cold raw vectors/states invalid")
            mask,nativemask,selected=masks(full,state,arm)
            seen={"actual":e["actual_input"],"layers":e["actual_layers"],
                  "expected":nativemask if arm==ARMS[0] else mask,"selected":selected}
            hashes=verify_input(model,batch,raw,pad,state,arm,emitted,full,seen,logits,e["chosen"],
                                previous=emitted[-1] if t else None)
            require(hashes==e["input_hashes"] and e["actual_mask_hash"]==tensor_hash(seen["expected"]) and
                    e["selected_cells"]==9*t and
                    e["selected_native_hash"]==tensor_hash(nativemask[selected]) and
                    e["selected_actual_hash"]==tensor_hash(seen["expected"][selected]) and
                    e["complement_native_hash"]==e["complement_actual_hash"]==tensor_hash(nativemask[~selected]),
                    "cold actual mask/complement differs")
            if arm==ARMS[0]:
                parity=[_trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=state["prefix_end"]+t,token_id=raw[i]["token_ids"][state["prefix_end"]+t],
                    role="cold_book_native") for i in range(4)]
                require(all(x["passed"] for x in parity) and e["chosen"]==state["native_expected_ids"][t],
                        "cold native source parity failed")
                native.append({"chosen":e["chosen"],"payload":payload})
            else:
                parity=[_trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=state["prefix_end"]+t,token_id=raw[i]["token_ids"][state["prefix_end"]+t],
                    role="cold_book_companion") for i in (0,2,3)]
                require(all(x["passed"] for x in parity),"cold companion source parity failed")
                if t<9:
                    old=native[t]["payload"]
                    require(max(float((payload["history"]-old["history"]).abs().max()),
                                float((payload["companions"]-old["companions"]).abs().max()),
                                max(float((logits[i]-old["logits"][i]).abs().max()) for i in (0,2,3)))<=TOL,
                            "cold history/companion differs")
                    if arm==ARMS[1]:
                        require(float((logits-old["logits"]).abs().max())<=TOL and
                                e["chosen"]==state["native_expected_ids"][t],"cold sham differs")
            v=logits[TARGET].double();top=torch.topk(v,2);probs=torch.softmax(v,-1)
            steps.append({"step":t,"chosen":e["chosen"],"top2":top.indices.tolist(),
                          "top2_logits":top.values.tolist(),"chosen_prob":float(probs[e["chosen"]]),
                          "raw":e["raw"]})
            emitted.append(e["chosen"]);logical+=1;actual+=1
            parsed=parse_row(emitted,special)
            require(parsed["stop"] is None if t<len(record["steps"])-1 else
                    parsed["stop"]==record["stop"],"cold parser/stop changed")
        require(emitted==record["emitted"],"cold emitted tokens changed")
        parsed=parse_row(emitted,special)
        desc=q.tokenizer.decode(parsed["description_ids"],skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        box=[v-151670 for v in parsed["box_ids"]] if len(parsed["box_ids"])==4 else None
        geometry=("valid" if box[0]<box[2] and box[1]<box[3] else "invalid") if box else "no_complete_box"
        arms.append({"arm":arm,"emitted":emitted,"stop":parsed["stop"],"description_ids":parsed["description_ids"],
                     "description":desc,"box":box,"geometry":geometry,"steps":steps})
    require(actual==r["counts"]["model_forwards"]==r["counts"]["vision_forwards"] and
            logical==r["counts"]["logical_tokens"] and
            r["counts"]["reused_t0"]==1 and actual<=33 and logical<=34,
            "cold actual/logical counts differ")
    result={"status":"candidate_cold_readback_passed","state":state_id,"receipt":bind(root/"receipt.json"),
            "arms":arms,"counts":r["counts"],"secondary_exact":arms[2]["emitted"]==state["masked_secondary_exact_expected_ids"]}
    write_new(root/"readback.json",result)
    print(json.dumps({"status":result["status"],"state":state_id,"counts":result["counts"],
                      "rows":{x["arm"]:x["emitted"] for x in arms},"secondary_exact":result["secondary_exact"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","run_row4","readback_row4","run_row3","readback_row3"))
    action=parser.parse_args().action
    if action=="preflight":preflight()
    elif action.startswith("run_"):run_state(STATES[0] if action=="run_row4" else STATES[1])
    else:readback(STATES[0] if action=="readback_row4" else STATES[1])
