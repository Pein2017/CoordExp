"""One fixed book B historical-extent factorial, full-prefix free-row replay."""
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

from probes.training_set_completion.recurrence_book_first_revisit import run as prior

REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-same-owner-extent"
PROTOCOL = UNIT / "extent-protocol.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/extent-preflight-v1.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-same-owner-extent/extent-v1")
SHAS = {PROTOCOL: "1dc907a4443794b3005d4b542dd7bf6ec2b5556134f73b5f02bcbbbbd1f7e0cd",
        ADMISSION: "a6f02b13d9e5677bb1a58a01a874f272c2fa22d52044125fa168fdd91e5ce790"}
ARMS = ("native", "identity_write_sham", "both", "x1_only", "y2_only")
FIRST, SECOND = ARMS[:3], ARMS[3:]
X1 = (151867,151867,151877,151877,151867)
Y2 = (152529,152529,152522,152529,152522)
NATIVE = (151646,2190,151647,151648,151877,152499,151940,152522,151649)
Q = (151646,2190,151647,151648,151926,152346,152058,152421,151649)
B_BOX, Q_BOX = (207,829,270,852), (256,676,388,751)
TARGET, PREFIX, PROMPT, TOL = 1,36,1362,2e-4
KEYS = ("input_ids","attention_mask","position_ids","cache_position")
MEDIA = ("pixel_values","image_grid_thw")
require, bind, write_new = prior.require, prior.bind, prior.write_new


def contract():
    for path,sha in SHAS.items():
        require(bind(path)["sha256"]==sha,f"frozen owner changed: {path}")
    a=json.loads(ADMISSION.read_text())
    require(a["status"]=="lead-admitted-finite-five-arm-package" and
            a["worker_thread"]=="01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"]=="gpt-6-sol" and a["worker_effort"]=="xhigh" and
            a["raw_output_root"]==str(OUT) and a["ordered_arms"]==list(ARMS) and
            a["max_model_forwards"]==a["max_vision_forwards"]==a["max_emitted_tokens"]==66 and
            a["treatment_reused_calls"]==0 and a["max_width"]==1413 and
            a["changed_raw_slots"]==[31,34] and a["changed_physical_slots"]==[1393,1396] and
            a["primary"]["B_reference_box"]==list(B_BOX) and
            a["primary"]["Q_reference_box"]==list(Q_BOX) and
            a["primary"]["region_target_min_iou"]==0.5 and
            a["primary"]["region_other_max_iou"]==0.1 and
            a["primary"]["boundary_guard"]==1e-6 and
            tuple(a["native_reference"]["tokens"])==NATIVE and
            tuple(a["numerical_reference_Q"]["tokens"])==Q,
            "finite admission/criterion changed")
    for name in ("protocol","cpu_protocol","cpu_acceptance","cpu_candidate","cpu_bindings",
                 "predecessor_acceptance","source_panel"):
        b=a[name];require(bind(b["path"])==b,f"bound dependency changed: {name}")
    for b in a["source_bindings"].values(): require(bind(b["path"])==b,"source changed")
    for item in a["historical_loader_path_crosswalk"]:
        m=bind(item["maintained_path"])
        require(m["sha256"]==item["sha256"]==item["historical"]["sha256"] and
                m["size_bytes"]==item["size_bytes"]==item["historical"]["size_bytes"],
                "maintained loader crosswalk changed")
    old=prior.contract()
    require(old["source_bindings"]==a["source_bindings"] and
            old["source_full_batch"]==a["source_full_batch"],"predecessor source differs")
    return a,old


def history(raw,arm,*,target=TARGET,slot=(31,34),x1=None,y2=None):
    require(arm in ARMS and target==TARGET and slot==(31,34),"wrong target/history slot")
    native=list(raw[TARGET]["token_ids"][:PREFIX])
    require(len(native)==PREFIX and native[27:36]==
            [151646,2190,151647,151648,151867,152499,151940,152529,151649],
            "native B record changed")
    j=ARMS.index(arm)
    require(x1 in (None,X1[j]) and y2 in (None,Y2[j]),"wrong fixed corner value")
    native[31]=X1[j];native[34]=Y2[j]
    return native


def inputs(model,batch,raw,pad,arm,emitted,*,previous=None,target=TARGET,slot=(31,34),x1=None,y2=None):
    t=len(emitted)
    require(0<=t<16 and (t==0 or emitted[-1]==previous),
            "wrong step/own greedy token")
    h=history(raw,arm,target=target,slot=slot,x1=x1,y2=y2)
    tails=prior._prefix_tokens(raw,PREFIX+t,pad)
    tails[TARGET]=h+list(emitted)
    full=prior.exact_history_inputs(model,batch.inputs,
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
    hashes={k:prior.tensor_hash(expected[k]) for k in KEYS+MEDIA}
    require(all(torch.equal(full[k],expected[k]) for k in KEYS+MEDIA) and actual==hashes and
            layers==list(range(28)) and chosen==int(torch.argmax(logits[TARGET]).item()),
            "actual source/history/own-prefix/position/mask/greedy differs")
    original=inputs(model,batch,raw,pad,"native",emitted,previous=previous)
    delta=torch.nonzero(full["input_ids"]!=original["input_ids"],as_tuple=False).tolist()
    wanted=([1,1393] if X1[ARMS.index(arm)]!=X1[0] else None,
            [1,1396] if Y2[ARMS.index(arm)]!=Y2[0] else None)
    require(delta==[x for x in wanted if x is not None] and
            all(torch.equal(full[k],original[k]) for k in ("attention_mask","position_ids","cache_position")+MEDIA),
            "additional historical/companion/mask/position change")
    return hashes,delta


def cpu_checks(batch,raw,pad,special):
    model=prior.ConfigOnlyRope();checks=[]
    require(prior.parse_row(list(NATIVE),special)["stop"]=="complete" and
            prior.parse_row(list(Q),special)["stop"]=="complete" and
            prior.parse_row([151645],special)["stop"]=="eos" and
            prior.parse_row([151649],special)["stop"]=="early_row_terminator" and
            prior.parse_row([151646,2190,151647,151648,152670],special)["stop"]=="malformed_coordinate" and
            prior.parse_row([151646,2190,151647,151648,152669],special)["stop"] is None and
            prior.parse_row([151646]+[2190]*15,special)["stop"]=="cap" and
            prior.parse_row([151646,2190,151647,151648,151670,151671,151672,151673,151670],special)["stop"]=="malformed_terminator",
            "free-row parser boundaries changed")
    checks.append("parser_boundaries")
    config=AutoConfig.from_pretrained(prior.BASE,local_files_only=True).text_config
    config._attn_implementation="sdpa"
    native_t0=inputs(model,batch,raw,pad,"native",[])
    for arm in ARMS:
        for t in (0,8 if arm in FIRST[:2] else 15):
            emitted=list(NATIVE[:t]) if t<=9 else [151646]+[2190]*(t-1)
            previous=emitted[-1] if t else None
            full=inputs(model,batch,raw,pad,arm,emitted,previous=previous)
            reference=create_causal_mask(config,torch.empty((*full["attention_mask"].shape,1)),
                full["attention_mask"],full["cache_position"],None,position_ids=full["position_ids"][0])
            require(torch.equal(reference,prior.native_4d(full["attention_mask"])),
                    "native installed SDPA Boolean mask differs")
            class Fake:
                seen=None
                def __call__(self,**kwargs):
                    self.seen={k:prior.tensor_hash(kwargs[k]) for k in KEYS+MEDIA}
                    v=torch.zeros((4,152670));v[TARGET,151646]=1
                    return SimpleNamespace(logits=v[:,None,:])
            fake=Fake();logits=caller(fake,full).logits[:,-1]
            verify_step(model,batch,raw,pad,arm,emitted,full,fake.seen,logits,151646,
                        list(range(28)),previous=previous)
            checks.append(f"actual_caller_receipt_{arm}_t{t}")
            if arm not in FIRST[:2] and t==0:
                require(prior.tensor_hash(full["input_ids"])!=prior.tensor_hash(native_t0["input_ids"]),
                        "false native t0 reuse")
                try:verify_step(model,batch,raw,pad,arm,emitted,full,
                                {**fake.seen,"input_ids":prior.tensor_hash(native_t0["input_ids"])},
                                logits,151646,list(range(28)))
                except ValueError:checks.append("reject_native_t0_reuse")
                else:raise AssertionError("native t0 reused")
            j=ARMS.index(arm)
            for label,kw in (("target",{"target":2}),("slot",{"slot":(31,35)}),
                             ("x1",{"x1":X1[j]+1}),("y2",{"y2":Y2[j]+1})):
                try:inputs(model,batch,raw,pad,arm,emitted,previous=previous,**kw)
                except ValueError:checks.append(f"reject_{label}_{arm}_t{t}")
                else:raise AssertionError(label)
            for label,key,index in (("other_history","input_ids",(TARGET,PROMPT+30)),
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
    q=prior.load_qwen_components_from_options(prior.QwenLoadOptions(base_model=str(prior.BASE),
        dtype="fp32",attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded model")
    batch,raw,trace,sr,planning=prior.source(q,old,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    checks=cpu_checks(batch,raw,pad,special)
    require(len(checks)>70 and a["source_full_batch"]["pixel_elements"]==int(batch.inputs["pixel_values"].numel()) and
            PROMPT+PREFIX+15==a["max_width"] and a["forecast_outer_gpu_seconds"]<500 and
            a["artifact_planning_envelope_bytes"]==2*1024**3 and
            a["prior_sequence_gpu_hours"]==0.3544195017876344,
            "CPU source/cost shape changed")
    # The accepted predecessor path is fixed by its prior admission, not this output root.
    old_readback=Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-book-first-revisit/native-read-v1/row4_first_B_revisit/readback.json")
    old_r=json.loads(old_readback.read_text())
    refs=[e["raw"] for e in old_r["arms"][0]["steps"]]
    require(len(refs)==9 and [e["chosen"] for e in old_r["arms"][0]["steps"]]==list(NATIVE) and
            all(bind(r["path"])==r for r in refs),"accepted nine reference vectors changed")
    earlier=json.loads(prior.PREFLIGHT.read_text())
    files=[Path(__file__)]+[Path(c["maintained"]["path"]) for c in earlier["captures"]]
    files.append(REPO/"probes/training_set_completion/artifacts.py")
    captures=[]
    for path in dict.fromkeys(files):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=prior.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_same_owner_extent.run"]
    p={"status":"cpu_qualified_before_gpu","admission":bind(ADMISSION),"protocol":bind(PROTOCOL),
       "producer":bind(Path(__file__)),"source_identity":sr["identity"],"input_identity":sr["input_identity"],
       "planning":planning,"pad":pad,"special":sorted(special),"cpu_checks":checks,
       "full_source_batch":a["source_full_batch"],"max_width":a["max_width"],
       "forecast_outer_seconds":a["forecast_outer_gpu_seconds"],
       "artifact_envelope_bytes":a["artifact_planning_envelope_bytes"],
       "accepted_native_readback":bind(old_readback),"accepted_native_vectors":refs,
       "captures":captures,"commands":{x:command+[x] for x in ("run_first","readback_first","run_second","readback_second")}}
    write_new(PREFLIGHT,p)
    print(json.dumps({"status":p["status"],"cpu_checks":len(checks),"captures":len(captures),
                      "max_width":p["max_width"],"forecast_outer_seconds":p["forecast_outer_seconds"]}))


def checked():
    a,old=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["admission"]==bind(ADMISSION) and
            p["producer"]==bind(Path(__file__)) and
            p["accepted_native_readback"]==bind(p["accepted_native_readback"]["path"]) and
            all(bind(r["path"])==r for r in p["accepted_native_vectors"]),"preflight/reference changed")
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
        q,identity=prior.load_model("untied",device)
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in p["source_identity"].items() if k!="loader_source"} and
                all(identity["loader_source"][k]==p["source_identity"]["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"]==a["historical_loader_path_crosswalk"][0]["maintained_path"],
                "effective model/maintained loader changed")
        model=q.model.eval();batch,raw,trace,sr,planning=prior.source(q,old,device)
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
            active["actual"]={k:prior.tensor_hash(kwargs[k]) for k in KEYS+MEDIA}
            require(active["actual"]=={k:prior.tensor_hash(active["full"][k]) for k in KEYS+MEDIA},
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
                    active.update(full=full,native_4d=prior.native_4d(full["attention_mask"]),
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
                           "actual_layers":active["layers"],"actual_mask_hash":prior.tensor_hash(active["native_4d"]),
                           "changed_history_physical_slots":delta,"chosen":chosen,
                           "internal_seconds":time.monotonic()-start}
                    if arm=="native":
                        entry["source_trace"]=[prior._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=PREFIX+t,
                            token_id=int(raw[i]["token_ids"][PREFIX+t]),role="extent_native") for i in range(4)]
                        ref=p["accepted_native_vectors"][t]
                        oldlogits=torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]
                        entry["accepted_full_vector_max_error"]=float((logits-oldlogits).abs().max())
                        require(all(x["passed"] for x in entry["source_trace"]) and
                                entry["accepted_full_vector_max_error"]<=TOL and chosen==NATIVE[t],
                                "native source/reference vector/greedy parity failed")
                    else:
                        entry["companion_source_trace"]=[prior._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=PREFIX+t,
                            token_id=int(raw[i]["token_ids"][PREFIX+t]),role="extent_companion")
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
                    parsed=prior.parse_row(emitted,special)
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
    q=prior.load_qwen_components_from_options(prior.QwenLoadOptions(base_model=str(prior.BASE),
        dtype="fp32",attn_implementation="sdpa",patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold readback loaded model")
    batch,raw,trace,sr,planning=prior.source(q,old,torch.device("cpu"))
    require(sr["input_identity"]==p["input_identity"] and planning==p["planning"],"cold source changed")
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    model=prior.ConfigOnlyRope();items=[];actual=0
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
                    e["actual_mask_hash"]==prior.tensor_hash(prior.native_4d(full["attention_mask"])),
                    "cold vectors/mask/states invalid")
            hashes,delta=verify_step(model,batch,raw,pad,arm,emitted,full,e["actual_input"],logits,
                                     e["chosen"],e["actual_layers"],previous=emitted[-1] if t else None)
            require(hashes==e["input_hashes"] and delta==e["changed_history_physical_slots"],
                    "cold actual full input/history write differs")
            if arm=="native":
                parity=[prior._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=PREFIX+t,token_id=int(raw[i]["token_ids"][PREFIX+t]),
                    role="cold_extent_native") for i in range(4)]
                ref=p["accepted_native_vectors"][t]
                oldlogits=torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]
                require(all(x["passed"] for x in parity) and
                        float((logits-oldlogits).abs().max())<=TOL and e["chosen"]==NATIVE[t],
                        "cold native source/reference parity failed")
            else:
                parity=[prior._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=PREFIX+t,token_id=int(raw[i]["token_ids"][PREFIX+t]),
                    role="cold_extent_companion") for i in (0,2,3)]
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
            parsed=prior.parse_row(emitted,special)
            require(parsed["stop"] is None if t<len(item["steps"])-1 else parsed["stop"]==item["stop"],
                    "cold parser/stop differs")
        require(emitted==item["emitted"],"cold own generated tokens differ")
        parsed=prior.parse_row(emitted,special)
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
