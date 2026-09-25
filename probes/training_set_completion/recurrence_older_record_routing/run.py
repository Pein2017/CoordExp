"""Fixed older-book record content/order at the original post-B-revisit prefix."""
from __future__ import annotations

import argparse
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

from probes.training_set_completion.recurrence_same_owner_extent import run as prior

book = prior.prior

REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-older-record-routing"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/older-record-preflight-v1.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-record-routing/attempt-001")
SHAS = {PROTOCOL: "6735d7b151cb0b3f23b131c38fa5dc9b6145e260ca7d51ae6270b8307f2f4b4d",
        ADMISSION: "c9f3aa466ce19a37f5ab56ea556722ecae2bd68704832160386c43986b909d4e"}
ARMS = ("native", "identity_write_sham", "earlier_A", "earlier_B1", "swap_earlier")
FIRST, SECOND = ARMS[:3], ARMS[3:]
A = (151859,152438,151924,152484)
B0 = (151867,152499,151940,152529)
B1 = (151877,152499,151940,152522)
ROWS = ((A,B0),(A,B0),(A,A),(A,B1),(B0,A))
CHANGED = ((),(),(31,32,33,34),(31,34),(22,23,24,25,31,32,33,34))
NATIVE = (151646,2190,151647,151648,151926,152346,152058,152421,151649)
B1_ROW = (151646,2190,151647,151648,*B1,151649)
B_BOX, Q_BOX = (207,829,270,852), (256,676,388,751)
TARGET, PREFIX, PROMPT, TOL = 1,45,1362,2e-4
KEYS = ("input_ids","attention_mask","position_ids","cache_position")
MEDIA = ("pixel_values","image_grid_thw")
require, bind, write_new = prior.require, prior.bind, prior.write_new


def contract():
    for path,sha in SHAS.items():
        require(bind(path)["sha256"]==sha,f"frozen owner changed: {path}")
    a=json.loads(ADMISSION.read_text())
    require(a["status"]=="lead-admitted-finite-older-record-routing" and
            a["worker_thread"]=="01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"]=="gpt-6-sol" and a["worker_effort"]=="xhigh" and
            a["raw_output_root"]==str(OUT) and a["ordered_arms"]==list(ARMS) and
            a["raw_prefix_end"]==PREFIX and a["latest_row_raw_span"]==[36,45] and
            a["writable_coordinate_raw_spans"]==[[22,26],[31,35]] and
            a["max_model_forwards"]==a["max_vision_forwards"]==a["max_emitted_tokens"]==66 and
            a["reused_calls"]==0 and a["max_width"]==1422 and
            a["primary"]["B_reference_box"]==list(B_BOX) and
            a["primary"]["Q_reference_box"]==list(Q_BOX) and
            a["primary"]["target_iou_min"]==0.5 and a["primary"]["other_iou_max"]==0.1 and
            a["primary"]["boundary_guard"]==1e-6 and
            tuple(a["native_reference"]["tokens"])==NATIVE and
            a["native_reference"]["kind"]=="original-source-trace-only; no pre-existing full vectors",
            "finite admission/criterion changed")
    for j,arm in enumerate(ARMS):
        entry=a["arms"][j]
        require(entry["arm"]==arm and entry["changed_raw_indices"]==list(CHANGED[j]) and
                entry["changed_full_input_indices"]==[[TARGET,PROMPT+i] for i in CHANGED[j]] and
                entry["max_tokens"]==(9 if j<2 else 16),"admitted arm/write changed")
    for name in ("protocol","predecessor_acceptance","predecessor_admission","source_panel"):
        b=a[name];require(bind(b["path"])==b,f"bound dependency changed: {name}")
    for b in a["source_bindings"].values():require(bind(b["path"])==b,"source changed")
    for item in a["historical_loader_path_crosswalk"]:
        m=bind(item["maintained_path"])
        require(m["sha256"]==item["sha256"]==item["historical"]["sha256"] and
                m["size_bytes"]==item["size_bytes"]==item["historical"]["size_bytes"],
                "maintained loader crosswalk changed")
    previous,book_admission=prior.contract()
    require(previous["source_bindings"]==a["source_bindings"] and
            previous["source_full_batch"]==a["source_full_batch"],"predecessor source differs")
    return a,book_admission

def history(raw,arm,*,target=TARGET,rows=((18,27),(27,36)),donor=None):
    require(arm in ARMS and target==TARGET and rows==((18,27),(27,36)),
            "wrong target/earlier row spans")
    original=list(raw[TARGET]["token_ids"][:PREFIX])
    require(len(original)==PREFIX and original[18:27]==[151646,2190,151647,151648,*A,151649] and
            original[27:36]==[151646,2190,151647,151648,*B0,151649] and
            original[36:45]==list(B1_ROW),"native A/B0/B1 records changed")
    j=ARMS.index(arm)
    require(donor is None or donor==ROWS[j],"wrong fixed donor pair")
    written=original.copy()
    written[22:26]=ROWS[j][0];written[31:35]=ROWS[j][1]
    require(written[36:45]==original[36:45] and
            all(written[k]==original[k] for k in range(PREFIX) if k not in CHANGED[j]) and
            (arm!="swap_earlier" or sorted(written)==sorted(original)),
            "latest B/other history or swap multiset changed")
    return written

def inputs(model,batch,raw,pad,arm,emitted,*,previous=None,target=TARGET,
           rows=((18,27),(27,36)),donor=None):
    t=len(emitted)
    require(0<=t<16 and (t==0 or emitted[-1]==previous),"wrong step/own greedy token")
    h=history(raw,arm,target=target,rows=rows,donor=donor)
    tails=book._prefix_tokens(raw,PREFIX+t,pad)
    tails[TARGET]=h+list(emitted)
    full=book.exact_history_inputs(model,batch.inputs,
        [list(p)+tail for p,tail in zip(batch.prompt_token_ids,tails,strict=True)],
        pad_token_id=pad,logits_to_keep=1)
    width=int(full["input_ids"].shape[1]);full["cache_position"]=torch.arange(width,device=full["input_ids"].device)
    require(width==PROMPT+PREFIX+t and full["attention_mask"].shape==(4,width) and
            full["position_ids"].shape==(3,4,width) and
            full["input_ids"][:,PROMPT:].tolist()==tails and
            all(tails[i]==raw[i]["token_ids"][:PREFIX+t] for i in (0,2,3)) and
            tails[TARGET][PREFIX:]==list(emitted) and
            all(torch.equal(full[k],batch.inputs[k]) for k in MEDIA),
            "full source/media/own-prefix shape changed")
    return full

def caller(model,full):
    return model(**full)


def verify_step(model,batch,raw,pad,arm,emitted,full,actual,logits,chosen,layers,*,previous=None):
    expected=inputs(model,batch,raw,pad,arm,emitted,previous=previous)
    hashes={k:book.tensor_hash(expected[k]) for k in KEYS+MEDIA}
    require(all(torch.equal(full[k],expected[k]) for k in KEYS+MEDIA) and actual==hashes and
            layers==list(range(28)) and chosen==int(torch.argmax(logits[TARGET]).item()),
            "actual source/history/own-prefix/position/mask/greedy differs")
    original=inputs(model,batch,raw,pad,"native",emitted,previous=previous)
    delta=torch.nonzero(full["input_ids"]!=original["input_ids"],as_tuple=False).tolist()
    wanted=[[TARGET,PROMPT+i] for i in CHANGED[ARMS.index(arm)]]
    require(delta==wanted and all(torch.equal(full[k],original[k]) for k in
            ("attention_mask","position_ids","cache_position")+MEDIA),
            "additional history/latest-B/companion/mask/position change")
    return hashes,delta

def cpu_checks(batch,raw,pad,special,a):
    model=book.ConfigOnlyRope();checks=[]
    require(book.parse_row(list(NATIVE),special)["stop"]=="complete" and
            book.parse_row(list(B1_ROW),special)["stop"]=="complete" and
            book.parse_row([151645],special)["stop"]=="eos" and
            book.parse_row([151649],special)["stop"]=="early_row_terminator" and
            book.parse_row([151646,2190,151647,151648,152670],special)["stop"]=="malformed_coordinate" and
            book.parse_row([151646,2190,151647,151648,152669],special)["stop"] is None and
            book.parse_row([151646]+[2190]*15,special)["stop"]=="cap" and
            book.parse_row([151646,2190,151647,151648,151670,151671,151672,151673,151670],special)["stop"]=="malformed_terminator",
            "free-row parser boundaries changed")
    checks.append("parser_boundaries")
    config=AutoConfig.from_pretrained(book.BASE,local_files_only=True).text_config
    config._attn_implementation="sdpa"
    native_t0=inputs(model,batch,raw,pad,"native",[])
    for j,arm in enumerate(ARMS):
        require(history(raw,arm)==a["arms"][j]["target_raw_history"],"admission history differs")
        for t in (0,8 if arm in FIRST[:2] else 15):
            emitted=list(NATIVE[:t]) if t<=9 else [151646]+[2190]*(t-1)
            previous=emitted[-1] if t else None
            full=inputs(model,batch,raw,pad,arm,emitted,previous=previous)
            reference=create_causal_mask(config,torch.empty((*full["attention_mask"].shape,1)),
                full["attention_mask"],full["cache_position"],None,position_ids=full["position_ids"][0])
            require(torch.equal(reference,book.native_4d(full["attention_mask"])),
                    "native installed SDPA Boolean mask differs")
            class Fake:
                seen=None
                def __call__(self,**kwargs):
                    self.seen={k:book.tensor_hash(kwargs[k]) for k in KEYS+MEDIA}
                    v=torch.zeros((4,152670));v[TARGET,151646]=1
                    return SimpleNamespace(logits=v[:,None,:])
            fake=Fake();logits=caller(fake,full).logits[:,-1]
            verify_step(model,batch,raw,pad,arm,emitted,full,fake.seen,logits,151646,
                        list(range(28)),previous=previous)
            checks.append(f"actual_caller_receipt_{arm}_t{t}")
            if arm not in FIRST[:2] and t==0:
                require(book.tensor_hash(full["input_ids"])!=book.tensor_hash(native_t0["input_ids"]),
                        "false native t0 reuse")
                try:verify_step(model,batch,raw,pad,arm,emitted,full,
                                {**fake.seen,"input_ids":book.tensor_hash(native_t0["input_ids"])},
                                logits,151646,list(range(28)))
                except ValueError:checks.append("reject_native_t0_reuse")
                else:raise AssertionError("native t0 reused")
            for label,kw in (("target",{"target":2}),("row",{"rows":((18,27),(28,37))}),
                             ("donor",{"donor":(B1,A)})):
                if label=="donor" and (B1,A)==ROWS[j]:kw={"donor":(A,B1)}
                try:inputs(model,batch,raw,pad,arm,emitted,previous=previous,**kw)
                except ValueError:checks.append(f"reject_{label}_{arm}_t{t}")
                else:raise AssertionError(label)
            for label,key,index in (("other_history","input_ids",(TARGET,PROMPT+20)),
                                    ("latest_B","input_ids",(TARGET,PROMPT+40)),
                                    ("companion","input_ids",(0,PROMPT+30)),
                                    ("position","position_ids",(0,TARGET,PROMPT+31)),
                                    ("mask","attention_mask",(TARGET,PROMPT+31)),
                                    ("media","pixel_values",(0,0))):
                bad={**full,key:full[key].clone()};bad[key][index]+=1
                try:verify_step(model,batch,raw,pad,arm,emitted,bad,fake.seen,logits,151646,
                                list(range(28)),previous=previous)
                except ValueError:checks.append(f"reject_{label}_{arm}_t{t}")
                else:raise AssertionError(label)
            if t:
                try:inputs(model,batch,raw,pad,arm,emitted,previous=-1)
                except ValueError:checks.append("reject_wrong_own_previous")
                else:raise AssertionError("own previous")
            try:verify_step(model,batch,raw,pad,arm,emitted,full,fake.seen,logits,151647,
                            list(range(28)),previous=previous)
            except ValueError:checks.append("reject_greedy")
            else:raise AssertionError("greedy")
    return checks

def preflight():
    a,old=contract();require(not PREFLIGHT.exists() and not OUT.exists(),"preflight/run already exists")
    q=book.load_qwen_components_from_options(book.QwenLoadOptions(base_model=str(book.BASE),
        dtype="fp32",attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded model")
    batch,raw,trace,sr,planning=book.source(q,old,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    checks=cpu_checks(batch,raw,pad,special,a)
    require(len(checks)>70 and a["source_full_batch"]["pixel_elements"]==int(batch.inputs["pixel_values"].numel()) and
            PROMPT+PREFIX+15==a["max_width"] and abs(a["forecast_outer_gpu_seconds"]-478.9288666723713)<1e-9 and
            a["artifact_planning_envelope_bytes"]==2*1024**3 and
            a["prior_sequence_gpu_hours"]==0.39875775146149434,
            "CPU source/cost shape changed")
    require(raw[TARGET]["token_ids"][PREFIX:PREFIX+9]==list(NATIVE) and
            all(int(trace["steps"][PREFIX+t]["raw_winners"][TARGET])==NATIVE[t]
                for t in range(9)),"native row5 source/trace changed")
    earlier=json.loads(prior.PREFLIGHT.read_text())
    files=[Path(__file__)]+[Path(c["maintained"]["path"]) for c in earlier["captures"]]
    files.append(REPO/"probes/training_set_completion/artifacts.py")
    captures=[]
    for path in dict.fromkeys(files):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=book.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_older_record_routing.run"]
    p={"status":"cpu_qualified_before_gpu","admission":bind(ADMISSION),"protocol":bind(PROTOCOL),
       "producer":bind(Path(__file__)),"source_identity":sr["identity"],"input_identity":sr["input_identity"],
       "planning":planning,"pad":pad,"special":sorted(special),"cpu_checks":checks,
       "full_source_batch":a["source_full_batch"],"max_width":a["max_width"],
       "forecast_outer_seconds":a["forecast_outer_gpu_seconds"],
       "artifact_envelope_bytes":a["artifact_planning_envelope_bytes"],
       "captures":captures,"commands":{x:command+[x] for x in ("run_first","readback_first","run_second","readback_second")}}
    write_new(PREFLIGHT,p)
    print(json.dumps({"status":p["status"],"cpu_checks":len(checks),"captures":len(captures),
                      "max_width":p["max_width"],"forecast_outer_seconds":p["forecast_outer_seconds"]}))


def checked():
    a,old=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["admission"]==bind(ADMISSION) and
            p["producer"]==bind(Path(__file__)) and
            p["protocol"]==bind(PROTOCOL),"preflight/source changed")
    for c in p["captures"]:
        require(bind(c["maintained"]["path"])==c["maintained"] and
                bind(c["capture"]["path"])==c["capture"],"direct source capture changed")
    return a,old,p


def run_block(block):
    a,old,p=checked();arms=FIRST if block=="first" else SECOND;root=OUT/block
    require(not root.exists(),"block already launched; no retry")
    if block=="second":
        first=json.loads((OUT/"first/receipt.json").read_text())
        cold=json.loads((OUT/"first/readback.json").read_text())
        require(first["status"]=="candidate_complete" and cold["status"]=="candidate_cold_readback_passed" and
                cold["receipt"]==bind(OUT/"first/receipt.json") and
                first["counts"]["model_forwards"]==first["counts"]["vision_forwards"]<=34,
                "first block cold qualification failed")
    root.mkdir(parents=True)
    start=time.monotonic();device=torch.device("cuda:0");handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"emitted_tokens":0,"reused_calls":0}
    r={"status":"running","block":block,"pid":os.getpid(),"started_unix":time.time(),
       "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
       "arms":[],"counts":counts}
    write_new(root/"launch.json",r)
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        q,identity=book.load_model("untied",device)
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in p["source_identity"].items() if k!="loader_source"} and
                all(identity["loader_source"][k]==p["source_identity"]["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"]==a["historical_loader_path_crosswalk"][0]["maintained_path"],
                "effective model/maintained loader changed")
        model=q.model.eval();batch,raw,trace,sr,planning=book.source(q,old,device)
        require(sr["input_identity"]==p["input_identity"] and planning==p["planning"],"GPU source differs")
        pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
        require(pad==p["pad"] and sorted(special)==p["special"],"tokenizer changed")
        layers=list(model.model.language_model.layers);attentions=[x.self_attn for x in layers]
        require(len(layers)==len(attentions)==28 and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual 28 text layers changed")
        active={}
        def top(_m,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=(34 if block=="first" else 32),"block model cap")
            active["actual"]={k:book.tensor_hash(kwargs[k]) for k in KEYS+MEDIA}
            require(active["actual"]=={k:book.tensor_hash(active["full"][k]) for k in KEYS+MEDIA},
                    "actual top-level input changed")
        def vision(_m,_args):
            counts["vision_forwards"]+=1
            require(counts["vision_forwards"]<=(34 if block=="first" else 32),"block vision cap")
        handles += [model.register_forward_pre_hook(top,with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(vision)]
        for i,x in enumerate(attentions):
            def hook(_m,_args,kwargs,layer=i):
                mask=kwargs.get("attention_mask")
                require(isinstance(mask,torch.Tensor) and mask.ndim==4 and
                        torch.equal(mask,active["native_4d"]),f"layer {layer} consumed wrong native mask")
                active["layers"].append(layer)
            handles.append(x.register_forward_pre_hook(hook,with_kwargs=True))
        for i,x in enumerate(layers):
            def state_hook(_m,_args,output,layer=i):
                v=output[0] if isinstance(output,tuple) else output
                require(isinstance(v,torch.Tensor) and v.ndim==3,"hidden states unavailable")
                active["history"].append(v[TARGET,PROMPT:PROMPT+PREFIX].detach().cpu().float())
                active["companions"].append(v[[0,2,3],-1].detach().cpu().float())
            handles.append(x.register_forward_hook(state_hook))
        r["effective_identity"]=identity
        with torch.inference_mode():
            for arm in arms:
                emitted=[];item={"arm":arm,"steps":[],"stop":None};r["arms"].append(item)
                for t in range(9 if arm in FIRST[:2] else 16):
                    previous=emitted[-1] if t else None
                    full=inputs(model,batch,raw,pad,arm,emitted,previous=previous)
                    active.update(full=full,native_4d=book.native_4d(full["attention_mask"]),
                                  actual=None,layers=[],history=[],companions=[])
                    out=caller(model,full);torch.cuda.synchronize(device)
                    require(len(active["layers"])==len(active["history"])==len(active["companions"])==28 and
                            active["actual"] is not None,"actual consumer/state incomplete")
                    logits=out.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape==(4,152670) and torch.isfinite(logits).all().item(),"invalid logits")
                    chosen=int(torch.argmax(logits[TARGET]).item())
                    hashes,delta=verify_step(model,batch,raw,pad,arm,emitted,full,active["actual"],logits,
                                             chosen,active["layers"],previous=previous)
                    payload={"arm":arm,"step":t,"logits":logits,
                             "history":torch.stack(active["history"]),
                             "companions":torch.stack(active["companions"])}
                    rawpath=root/f"{arm}-step{t}.pt";torch.save(payload,rawpath)
                    inputpath=root/f"inputs-{arm}-step{t}.json"
                    write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                    entry={"step":t,"raw":bind(rawpath),"inputs":bind(inputpath),
                           "input_hashes":hashes,"actual_input":active["actual"],
                           "actual_layers":active["layers"],"actual_mask_hash":book.tensor_hash(active["native_4d"]),
                           "changed_history_physical_slots":delta,"chosen":chosen,
                           "internal_seconds":time.monotonic()-start}
                    if arm=="native":
                        entry["source_trace"]=[book._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=PREFIX+t,
                            token_id=int(raw[i]["token_ids"][PREFIX+t]),role="older_record_native") for i in range(4)]
                        require(all(x["passed"] for x in entry["source_trace"]) and chosen==NATIVE[t],
                                "native original source trace/greedy failed")
                    else:
                        entry["companion_source_trace"]=[book._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=PREFIX+t,
                            token_id=int(raw[i]["token_ids"][PREFIX+t]),role="older_record_companion")
                            for i in (0,2,3)]
                        require(all(x["passed"] for x in entry["companion_source_trace"]),
                                "active companion source trace changed")
                        if t<9:
                            nr=(r["arms"][0]["steps"][t] if block=="first" else
                                json.loads((OUT/"first/receipt.json").read_text())["arms"][0]["steps"][t])
                            oldpayload=torch.load(nr["raw"]["path"],map_location="cpu",weights_only=True)
                            entry["companion_state_error"]=float((payload["companions"]-oldpayload["companions"]).abs().max())
                            entry["companion_vector_error"]=max(float((logits[i]-oldpayload["logits"][i]).abs().max()) for i in (0,2,3))
                            require(max(entry["companion_state_error"],entry["companion_vector_error"])<=TOL,
                                    "companion states/vectors changed")
                            if arm=="identity_write_sham":
                                entry["sham_full_vector_error"]=float((logits-oldpayload["logits"]).abs().max())
                                entry["sham_history_state_error"]=float((payload["history"]-oldpayload["history"]).abs().max())
                                require(max(entry["sham_full_vector_error"],entry["sham_history_state_error"])<=TOL and
                                        chosen==NATIVE[t],"identity-write sham differs")
                    item["steps"].append(entry);emitted.append(chosen);counts["emitted_tokens"]+=1
                    require(counts["emitted_tokens"]<=(34 if block=="first" else 32),"block token cap")
                    parsed=book.parse_row(emitted,special)
                    item["emitted"]=list(emitted);item["stop"]=parsed["stop"]
                    write_new(root/f"checkpoint-{arm}-{t}.json",r)
                    if parsed["stop"] is not None:break
                require(item["stop"] is not None,"arm failed to stop by cap")
                if arm in FIRST[:2]:
                    require(item["stop"]=="complete" and item["emitted"]==list(NATIVE),
                            "native/sham qualification failed")
        r["status"]="candidate_complete"
    except BaseException as exc:
        r["status"]="technical_invalid"
        r["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    finally:
        for h in handles:h.remove()
        if torch.cuda.is_available():torch.cuda.synchronize(device)
        r["counts"]=dict(counts)
        r["internal_seconds"]=time.monotonic()-start
        r["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        r["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        r["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        r["artifact_bytes_before_receipt"]=sum(x.stat().st_size for x in root.rglob("*") if x.is_file())
        r["terminal_pid"]=os.getpid()
        write_new(root/"receipt.json",r)
    print(json.dumps({"status":r["status"],"block":block,"counts":counts,
                      "internal_seconds":r["internal_seconds"],"failure":r.get("failure",{}).get("message")}))
    require(r["status"]=="candidate_complete","technical failure; no retry")


def iou(x,y):
    w=max(0,min(x[2],y[2])-max(x[0],y[0]));h=max(0,min(x[3],y[3])-max(x[1],y[1]));v=w*h
    return v/((x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-v)


def region(parsed):
    box=[x-151670 for x in parsed["box_ids"]] if len(parsed["box_ids"])==4 else None
    valid=bool(box and box[0]<box[2] and box[1]<box[3])
    if not valid:return {"box":box,"geometry":"invalid" if box else "no_complete_box",
                         "B_iou":None,"Q_iou":None,"q":None,"region":"not_applicable"}
    b=iou(box,B_BOX);q=iou(box,Q_BOX)
    if parsed["stop"]!="complete" or parsed["description_ids"]!=[2190]:label="noncanonical"
    elif any(abs(z-v)<=1e-6 for z in (b,q) for v in (0.5,0.1)):label="boundary_HOLD"
    elif b>=0.5 and q<=0.1:label="B"
    elif q>=0.5 and b<=0.1:label="Q"
    else:label="neither"
    return {"box":box,"geometry":"valid","B_iou":b,"Q_iou":q,"q":q-b,"region":label}


def readback(block):
    a,old,p=checked();root=OUT/block;r=json.loads((root/"receipt.json").read_text())
    arms=FIRST if block=="first" else SECOND
    require(r["status"]=="candidate_complete" and r["block"]==block and
            [x["arm"] for x in r["arms"]]==list(arms) and not (root/"readback.json").exists(),
            "receipt/readback state changed")
    q=book.load_qwen_components_from_options(book.QwenLoadOptions(base_model=str(book.BASE),
        dtype="fp32",attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold readback loaded model")
    batch,raw,trace,sr,planning=book.source(q,old,torch.device("cpu"))
    require(sr["input_identity"]==p["input_identity"] and planning==p["planning"],"cold source changed")
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    model=book.ConfigOnlyRope();items=[];actual=0
    native=(None if block=="first" else json.loads((OUT/"first/receipt.json").read_text()))
    for item in r["arms"]:
        arm=item["arm"];emitted=[];steps=[]
        for t,e in enumerate(item["steps"]):
            require(e["step"]==t and bind(e["raw"]["path"])==e["raw"] and
                    bind(e["inputs"]["path"])==e["inputs"],"raw/input binding changed")
            vals=json.loads(Path(e["inputs"]["path"]).read_text())
            full={k:torch.tensor(v,dtype=torch.long) for k,v in vals.items()}
            full.update({k:batch.inputs[k] for k in MEDIA})
            payload=torch.load(e["raw"]["path"],map_location="cpu",weights_only=True)
            logits=payload["logits"]
            require(payload["arm"]==arm and payload["step"]==t and
                    logits.shape==(4,152670) and torch.isfinite(logits).all().item() and
                    payload["history"].shape[0]==payload["companions"].shape[0]==28 and
                    e["actual_mask_hash"]==book.tensor_hash(book.native_4d(full["attention_mask"])),
                    "cold vectors/mask/states invalid")
            hashes,delta=verify_step(model,batch,raw,pad,arm,emitted,full,e["actual_input"],logits,
                                     e["chosen"],e["actual_layers"],previous=emitted[-1] if t else None)
            require(hashes==e["input_hashes"] and delta==e["changed_history_physical_slots"],
                    "cold actual full input/history write differs")
            if arm=="native":
                parity=[book._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=PREFIX+t,token_id=int(raw[i]["token_ids"][PREFIX+t]),
                    role="cold_older_record_native") for i in range(4)]
                require(all(x["passed"] for x in parity) and e["chosen"]==NATIVE[t],
                        "cold native original source trace/greedy failed")
            else:
                parity=[book._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=PREFIX+t,token_id=int(raw[i]["token_ids"][PREFIX+t]),
                    role="cold_older_record_companion") for i in (0,2,3)]
                require(all(x["passed"] for x in parity),"cold companion source parity failed")
                if t<9:
                    nr=(r["arms"][0]["steps"][t] if block=="first" else native["arms"][0]["steps"][t])
                    oldpayload=torch.load(nr["raw"]["path"],map_location="cpu",weights_only=True)
                    require(max(float((payload["companions"]-oldpayload["companions"]).abs().max()),
                                max(float((logits[i]-oldpayload["logits"][i]).abs().max()) for i in (0,2,3)))<=TOL,
                            "cold companion states/vectors differ")
                    if arm=="identity_write_sham":
                        require(float((logits-oldpayload["logits"]).abs().max())<=TOL and
                                float((payload["history"]-oldpayload["history"]).abs().max())<=TOL,
                                "cold sham full vector/history differs")
            v=logits[TARGET].double();top=torch.topk(v,2);prob=torch.softmax(v,-1)
            steps.append({"step":t,"chosen":e["chosen"],"top2":top.indices.tolist(),
                          "top2_logits":top.values.tolist(),"chosen_prob":float(prob[e["chosen"]]),"raw":e["raw"]})
            emitted.append(e["chosen"]);actual+=1
            parsed=book.parse_row(emitted,special)
            require(parsed["stop"] is None if t<len(item["steps"])-1 else parsed["stop"]==item["stop"],
                    "cold parser/stop differs")
        require(emitted==item["emitted"],"cold own generated tokens differ")
        parsed=book.parse_row(emitted,special)
        info=region(parsed)
        items.append({"arm":arm,"emitted":emitted,"stop":parsed["stop"],
                      "description_ids":parsed["description_ids"],
                      "description":q.tokenizer.decode(parsed["description_ids"],skip_special_tokens=False,
                         clean_up_tokenization_spaces=False),**info,"steps":steps})
    require(actual==r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==
            r["counts"]["emitted_tokens"] and r["counts"]["reused_calls"]==0 and
            actual<=(34 if block=="first" else 32),"cold counts/reuse differ")
    result={"status":"candidate_cold_readback_passed","block":block,"receipt":bind(root/"receipt.json"),
            "arms":items,"counts":r["counts"],"admission":bind(ADMISSION)}
    write_new(root/"readback.json",result)
    print(json.dumps({"status":result["status"],"block":block,"counts":result["counts"],
                      "rows":{x["arm"]:x["emitted"] for x in items},
                      "regions":{x["arm"]:x["region"] for x in items}}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","run_first","readback_first","run_second","readback_second"))
    action=parser.parse_args().action
    if action=="preflight":preflight()
    elif action.startswith("run_"):run_block("first" if action=="run_first" else "second")
    else:readback("first" if action=="readback_first" else "second")
