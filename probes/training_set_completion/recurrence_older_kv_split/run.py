"""One finite older-record K/V split at the accepted AF/FF common header."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_contextual_kv_carrier import run as car
from src.artifacts.source_provenance import preserve_source


ROOT=Path(__file__).resolve().parents[3]
UNIT=ROOT/"research/experiments/2026-09-24-recurrence-older-kv-split"
PROTOCOL,ADMISSION=UNIT/"unit.md",UNIT/"lead-admission-v1.json"
PREFLIGHT=UNIT/"supporting/attempt-001-preflight.json"
OUT=Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-kv-split/attempt-001")
SHAS={PROTOCOL:"7b4eb5bf1ba144d468440c84960ccfc8b8b73e0db4ed5cafb1fda9c6561e9fe0",
      ADMISSION:"619e7be923579a0225ac3e9dd3ce4f76a65c9c8b0207094643ed7387dccc3620"}
NAMES=("full_AF","full_FF","prefill_AF","prefill_FF","anchor_AF","anchor_FF",
       "sham_AF","sham_FF","joint_AF_from_FF","joint_FF_from_AF",
       "V_AF_from_FF","V_FF_from_AF","K_AF_from_FF","K_FF_from_AF")
TARGET,WIDTH,END,LAYERS,HEADS,DIM,TOL=2,1380,1384,28,8,128,2e-4
OLDER,LATEST=(1362,1371),(1371,1380)
bind,require,write_new=car.bind,car.require,car.write_new
cache_digest=car.cache_digest


def contract():
    for path,sha in SHAS.items():require(bind(path)["sha256"]==sha,"frozen contract changed")
    a=json.loads(ADMISSION.read_text())
    require(a["status"]=="lead-admitted-fourteen-calls-after-CPU-qualification" and
            a["worker_thread"]=="01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"]=="gpt-6-sol" and a["worker_effort"]=="xhigh" and
            [x["name"] for x in a["cells"]]==list(NAMES) and
            [x["vision"] for x in a["cells"]]==[1]*4+[0]*10 and
            a["patch_physical_span"]==list(OLDER) and
            a["untouched_latest_physical_span"]==list(LATEST) and
            a["max_model_forwards"]==14 and a["max_vision_forwards"]==4 and
            a["max_generated_tokens"]==0 and
            a["qualification"]["logit_max_abs_all_four"]==TOL and
            a["criteria"]["baseline_minimum"]==1e-6 and
            a["criteria"]["numerical_boundary_guard"]==1e-6 and
            a["owned_paths"][-1]==str(OUT),"finite older K/V admission changed")
    for key in ("protocol","predecessor_acceptance","predecessor_admission",
                "predecessor_verification","predecessor_receipt","predecessor_readback",
                "predecessor_producer","source_panel"):
        require(bind(a[key]["path"])==a[key],f"bound {key} changed")
    for key in ("raw","trace","runtime_receipt","image"):
        require(bind(a["source_bindings"][key]["path"])==a["source_bindings"][key],
                f"original {key} changed")
    require(bind(a["loader_crosswalk"]["maintained"]["path"])==
            a["loader_crosswalk"]["maintained"],"maintained loader changed")
    for group in ("saved_full_vector_references","saved_full_inputs","anchor_references"):
        for value in a[group].values():require(bind(value["path"])==value,f"saved {group} changed")
    for value in a["joint_references"].values():
        for key in ("vector","input","consumer"):
            require(bind(value[key]["path"])==value[key],f"joint {key} changed")
    return a


def axes(cell):
    return [cell[x] for x in ("base_origin","older_K_origin","older_V_origin","latest_KV_origin")]


def record_guard(cells,admitted):
    require(isinstance(cells,list) and len(cells)==14,"cold cell list/count changed")
    for i,cell in enumerate(cells):
        require(isinstance(cell,dict) and cell.get("call")==i+1 and
                cell.get("name")==NAMES[i] and
                cell.get("origins")==axes(admitted[i]) and
                isinstance(cell.get("consumer"),dict) and
                isinstance(cell.get("input"),dict),
                "cold cell order/origin/container changed")


def older_layer_selected(layer,base_row,donor_row,*,base_origin,donor_origin,mode,
                         older=OLDER,latest=LATEST,target=TARGET):
    require(base_origin in ("AF","FF") and donor_origin in ("AF","FF") and
            mode in ("anchor","sham","joint","V","K") and target==TARGET and
            older[1]-older[0]==latest[1]-latest[0]==9 and older[1]==latest[0],
            "wrong older K/V consumer mode")
    result={}
    for name in ("keys","values"):
        value=getattr(layer,name)
        got={"prompt":car.base.tensor_hash(value[target,:,:older[0],:]),
             "older":car.base.tensor_hash(value[target,:,older[0]:older[1],:]),
             "latest":car.base.tensor_hash(value[target,:,latest[0]:latest[1],:]),
             "companions":car.base.tensor_hash(value[[0,1,3]])}
        for span in ("prompt","latest","companions"):
            require(got[span]==base_row[name][span],f"unselected {name} {span} changed")
        source=(donor_row if (mode in ("sham","joint") or
                (mode=="K" and name=="keys") or (mode=="V" and name=="values")) else None)
        wanted=car.base.tensor_hash(source["older"][name]) if source is not None else base_row[name]["older"]
        require(got["older"]==wanted,f"selected/unselected older {name} changed")
        result[name]=got
    return result


@contextmanager
def older_patch(cache,donor_blocks,*,mode,base_origin,donor_origin,original_digest,
                width=WIDTH,older=OLDER,latest=LATEST,target=TARGET):
    """Older nine-position K and/or V edit, suffix, crop and finally restore."""
    with torch.inference_mode():
        require(mode in ("sham","joint","V","K") and
                base_origin in ("AF","FF") and donor_origin in ("AF","FF") and
                (base_origin==donor_origin)==(mode=="sham") and
                donor_blocks["origin"]==donor_origin and target==TARGET and
                older[1]-older[0]==9 and latest[0]==older[1] and latest[1]==width and
                len(cache.layers)==len(donor_blocks["layers"])==LAYERS and
                cache.get_seq_length()==width and cache_digest(cache)==original_digest,
                "wrong older patch mode/base/donor/target/span/cache")
        saved=[]
        try:
            for i,layer in enumerate(cache.layers):
                old={};saved.append((layer,old))
                for name in ("keys","values"):
                    if mode=="K" and name=="values" or mode=="V" and name=="keys":continue
                    value=getattr(layer,name)
                    donor=donor_blocks["layers"][i]["older"][name]
                    require(donor.shape==(HEADS,9,DIM) and torch.isfinite(donor).all().item(),
                            f"invalid older {name} donor")
                    old[name]=value[target,:,older[0]:older[1],:].clone()
                    value[target,:,older[0]:older[1],:].copy_(donor.to(value.device))
            yield
        finally:
            try:cache.crop(width)
            finally:
                for layer,old in saved:
                    for name,value in old.items():
                        getattr(layer,name)[target,:,older[0]:older[1],:].copy_(value)
            require(cache.get_seq_length()==width and cache_digest(cache)==original_digest,
                    "older K/V crop or finally restoration failed")


@contextmanager
def scope(cache,digest,*,mode,base_origin,donor_origin,donor_blocks):
    if mode=="anchor":
        with car.suffix_scope(cache,digest,base_origin=base_origin):yield
    else:
        with older_patch(cache,donor_blocks,mode=mode,base_origin=base_origin,
                         donor_origin=donor_origin,original_digest=digest):yield


def cell_mode(name):
    if name.startswith("anchor"):return "anchor"
    if name.startswith("sham"):return "sham"
    if name.startswith("joint"):return "joint"
    if name.startswith("V_"):return "V"
    if name.startswith("K_"):return "K"
    raise ValueError("unknown older suffix cell")


def cpu_older_checks(a):
    width,older,latest=22,(4,13),(13,22)
    caches={}
    for origin in ("AF","FF"):
        cache=DynamicCache()
        for i in range(LAYERS):
            k=torch.zeros((4,HEADS,width,DIM),dtype=torch.float32)
            v=torch.zeros_like(k)
            for tensor,shift in ((k,0),(v,100)):
                tensor[TARGET,:,older[0]:older[1],:]=i+1+shift+(0 if origin=="AF" else 20)
                tensor[TARGET,:,latest[0]:latest[1],:]=i+2+shift+(0 if origin=="AF" else 30)
            cache.update(k,v,i)
        caches[origin]=cache
    snapshots={o:car.blocks(c,o,older=older,latest=latest) for o,c in caches.items()}
    origins={o:car.segments(c,width=width,older=older,latest=latest) for o,c in caches.items()}
    digests={o:cache_digest(c) for o,c in caches.items()}
    checks=[]
    for base_origin,donor_origin,mode in (
            ("AF","AF","sham"),("FF","FF","sham"),
            ("AF","FF","joint"),("FF","AF","joint"),
            ("AF","FF","V"),("FF","AF","V"),
            ("AF","FF","K"),("FF","AF","K")):
        cache=caches[base_origin]
        context=older_patch(cache,snapshots[donor_origin],mode=mode,
                            base_origin=base_origin,donor_origin=donor_origin,
                            original_digest=digests[base_origin],width=width,
                            older=older,latest=latest)
        with context:
            for i,layer in enumerate(cache.layers):
                older_layer_selected(layer,origins[base_origin][i],
                    snapshots[donor_origin]["layers"][i],base_origin=base_origin,
                    donor_origin=donor_origin,mode=mode,older=older,latest=latest)
                zeros=torch.zeros((4,HEADS,4,DIM),dtype=torch.float32)
                cache.update(zeros,zeros,i)
        require(cache_digest(cache)==digests[base_origin],"normal fixture restore failed")
        checks.append(f"actual_{mode}_{base_origin}_from_{donor_origin}_all28_restore")
    try:
        with older_patch(caches["AF"],snapshots["FF"],mode="joint",base_origin="AF",
                         donor_origin="FF",original_digest=digests["AF"],width=width,
                         older=older,latest=latest):
            for i in range(LAYERS):
                zeros=torch.zeros((4,HEADS,4,DIM),dtype=torch.float32)
                caches["AF"].update(zeros,zeros,i)
            raise RuntimeError("forced older patch body failure")
    except RuntimeError as exc:
        require(str(exc)=="forced older patch body failure","wrong forced failure")
    require(cache_digest(caches["AF"])==digests["AF"],"forced exception restore failed")
    checks.append("forced_body_exception_crop_and_older_KV_restore")
    for label,kw in (
        ("wrong_base",{"original_digest":digests["FF"]}),
        ("wrong_donor",{"donor_origin":"AF"}),
        ("wrong_target",{"target":1}),
        ("wrong_span",{"older":(3,12)}),
        ("wrong_mode_axis",{"mode":"X"})):
        args={"mode":"joint","base_origin":"AF","donor_origin":"FF",
              "original_digest":digests["AF"],"width":width,
              "older":older,"latest":latest}
        args.update(kw)
        try:
            with older_patch(caches["AF"],snapshots["FF"],**args):pass
        except (ValueError,AssertionError):checks.append(f"reject_{label}")
        else:raise AssertionError(f"accepted {label}")
    with older_patch(caches["AF"],snapshots["FF"],mode="joint",base_origin="AF",
                     donor_origin="FF",original_digest=digests["AF"],width=width,
                     older=older,latest=latest):
        layer=caches["AF"].layers[0]
        for name in ("keys","values"):
            t=getattr(layer,name)
            t[TARGET,:,older[0]:older[1],:].copy_(snapshots["AF"]["layers"][0]["older"][name])
            try:older_layer_selected(layer,origins["AF"][0],snapshots["FF"]["layers"][0],
                                     base_origin="AF",donor_origin="FF",mode="joint",
                                     older=older,latest=latest)
            except ValueError:checks.append(f"reject_missing_joint_{name}")
            else:raise AssertionError(f"accepted missing joint {name}")
            t[TARGET,:,older[0]:older[1],:].copy_(snapshots["FF"]["layers"][0]["older"][name])
        for name in ("keys","values"):
            t=getattr(layer,name)
            t[TARGET,:,latest[0],:].add_(1)
            try:older_layer_selected(layer,origins["AF"][0],snapshots["FF"]["layers"][0],
                                     base_origin="AF",donor_origin="FF",mode="joint",
                                     older=older,latest=latest)
            except ValueError:checks.append(f"reject_latest_{name}_mutation")
            else:raise AssertionError(f"accepted latest {name} mutation")
            t[TARGET,:,latest[0],:].sub_(1)
        layer.values[0,:,0,:].add_(1)
        try:older_layer_selected(layer,origins["AF"][0],snapshots["FF"]["layers"][0],
                                 base_origin="AF",donor_origin="FF",mode="joint",
                                 older=older,latest=latest)
        except ValueError:checks.append("reject_companion_mutation")
        else:raise AssertionError("accepted companion mutation")
        layer.values[0,:,0,:].sub_(1)
        layer.keys[TARGET,:,older[0],:].add_(1)
        try:older_layer_selected(layer,origins["AF"][0],snapshots["FF"]["layers"][0],
                                 base_origin="AF",donor_origin="FF",mode="joint",
                                 older=older,latest=latest)
        except ValueError:checks.append("reject_partial_older_K")
        else:raise AssertionError("accepted partial older K")
        layer.keys[TARGET,:,older[0],:].sub_(1)
    require(cache_digest(caches["AF"])==digests["AF"],"mutation fixture failed restoration")
    for mode,name in (("V","keys"),("K","values")):
        with older_patch(caches["AF"],snapshots["FF"],mode=mode,base_origin="AF",
                         donor_origin="FF",original_digest=digests["AF"],width=width,
                         older=older,latest=latest):
            layer=caches["AF"].layers[0];t=getattr(layer,name)
            t[TARGET,:,older[0]:older[1],:].copy_(snapshots["FF"]["layers"][0]["older"][name])
            try:older_layer_selected(layer,origins["AF"][0],snapshots["FF"]["layers"][0],
                                     base_origin="AF",donor_origin="FF",mode=mode,
                                     older=older,latest=latest)
            except ValueError:checks.append(f"reject_{mode}_extra_unselected_{name}")
            else:raise AssertionError("accepted unselected axis write")
            t[TARGET,:,older[0]:older[1],:].copy_(snapshots["AF"]["layers"][0]["older"][name])
    stub={"call":1,"name":NAMES[0],"origins":axes(a["cells"][0]),"consumer":{},"input":{}}
    good=[{**stub,"call":i+1,"name":NAMES[i],"origins":axes(a["cells"][i])}
          for i in range(14)]
    record_guard(json.loads(json.dumps(good)),a["cells"])
    checks.append("serialized_list_guard_green_all14")
    for label,bad in (
        ("swapped",good[:12]+[good[13],good[12]]),
        ("dropped",good[:-1]),("extra",good+[good[-1]]),
        ("wrong_axis",good[:10]+[{**good[10],"origins":["AF","FF","FF","AF"]}]+good[11:]),
        ("wrong_container",{"cells":good})):
        try:record_guard(json.loads(json.dumps(bad)),a["cells"])
        except ValueError:checks.append(f"serialized_guard_reject_{label}")
        else:raise AssertionError(f"serialized guard accepted {label}")
    return checks


def cpu_checks(q,batch,raw,pad,a):
    source_checks,fulls=car.cpu_checks(q,batch,raw,pad,a)
    checks=cpu_older_checks(a)+source_checks
    suffix=car.split_inputs(fulls["AF"],object(),"suffix")
    for label,field,index in (("target_header","input_ids",(TARGET,0)),
                              ("companion","input_ids",(1,0)),
                              ("position","position_ids",(0,TARGET,0)),
                              ("mask","attention_mask",(1,WIDTH)),
                              ("cache_slot","cache_position",(0,))):
        bad=dict(suffix);bad[field]=suffix[field].clone();bad[field][index]+=1
        try:car.verify_actual_input(bad,suffix,media=False)
        except ValueError:checks.append(f"older_actual_caller_reject_{label}")
        else:raise AssertionError(f"accepted {label} caller mutation")
    mask=car.expected_suffix_mask(fulls["AF"])
    bad=mask.clone();bad[TARGET,0,0,OLDER[0]]=~bad[TARGET,0,0,OLDER[0]]
    try:car.verify_attention(bad,mask)
    except ValueError:checks.append("older_actual_mask_reject_wrong_key")
    else:raise AssertionError("accepted wrong older key mask")
    return checks,fulls


def preflight():
    a=contract();require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),
                         "attempt already prepared")
    q=car.base.load_qwen_components_from_options(car.base.QwenLoadOptions(
        base_model=str(car.base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded language model")
    batch,raw,_,sr,_=car.prior.source(q,a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id)
    checks,fulls=cpu_checks(q,batch,raw,pad,a)
    require(len(checks)>=55 and int(batch.inputs["pixel_values"].numel())==24502272,
            "actual older caller/source CPU checks incomplete")
    paths=[Path(__file__),Path(car.__file__),Path(car.prior.__file__),Path(car.base.__file__),
           Path(inspect.getfile(DynamicCache)),Path(inspect.getfile(modeling_qwen3_vl)),
           Path(inspect.getfile(sdpa_attention)),Path(preserve_source.__code__.co_filename)]
    old_pre=json.loads(car.PREFLIGHT.read_text())
    paths += [Path(c["maintained"]["path"]) for c in old_pre["source_captures"]]
    captures=[]
    for path in dict.fromkeys(paths):
        rel=path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("installed")/path.name
        saved=preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_older_kv_split.run"]
    forecast=2*14/10*a["measured_predecessor"]["outer_seconds"]
    raw_forecast=2*14/10*a["measured_predecessor"]["raw_bytes"]
    require(abs(forecast-a["planning_outer_seconds"])<1e-6 and
            raw_forecast<a["artifact_planning_bytes"],"frozen cost/bytes forecast changed")
    packet={"status":"cpu_qualified_before_gpu","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"producer":bind(Path(__file__)),
            "source_identity":sr["identity"],"input_identity":sr["input_identity"],
            "request_ids":list(batch.request_ids),"pad_id":pad,
            "pixel_elements":int(batch.inputs["pixel_values"].numel()),
            "shapes":{"full":[4,END],"prefill":[4,WIDTH],"suffix":[4,4],
                      "suffix_mask":[4,1,4,END]},"checks":checks,
            "full_inputs":{o:{name:car.base.tensor_hash(v) for name,v in full.items()
                              if isinstance(v,torch.Tensor)} for o,full in fulls.items()},
            "source_captures":captures,"forecast_outer_seconds":forecast,
            "raw_forecast_bytes":raw_forecast,
            "commands":{"preflight":command+["preflight"],"run":command+["run"],
                        "gpu_child":command+["gpu"],"readback":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),
                      "captures":len(captures),"forecast_outer_seconds":forecast,
                      "raw_forecast_bytes":raw_forecast}))


def checked():
    a=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and
            p["protocol"]==bind(PROTOCOL) and p["admission"]==bind(ADMISSION) and
            p["producer"]==bind(Path(__file__)),"preflight/producer binding changed")
    for c in p["source_captures"]:
        require(bind(c["maintained"]["path"])==c["maintained"] and
                bind(c["capture"]["path"])==c["capture"],"captured direct source changed")
    return a,p


def gpu_child():
    a,p=checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
            "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();device=torch.device("cuda:0");handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"generated_tokens":0}
    receipt={"status":"running","pid":os.getpid(),"begun_unix":time.time(),
             "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),
             "producer":bind(Path(__file__)),"counts":counts,"cells":[]}
    write_new(OUT/"launch.json",receipt)
    active={}
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q,identity=car.base.load_model("untied",device)
        expected=p["source_identity"]
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in expected.items() if k!="loader_source"} and
                all(identity["loader_source"][k]==expected["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"]==a["loader_crosswalk"]["maintained"]["path"],
                "effective model/loader changed")
        model=q.model.eval()
        batch,raw,trace,sr,_=car.prior.source(q,a,device)
        require(sr["input_identity"]==p["input_identity"] and
                int(q.tokenizer.pad_token_id)==p["pad_id"],"GPU source/tokenizer changed")
        fulls={o:car.make_full(model,batch,raw,p["pad_id"],o) for o in ("AF","FF")}
        require({o:{name:car.base.tensor_hash(v) for name,v in full.items()
                    if isinstance(v,torch.Tensor)} for o,full in fulls.items()}==p["full_inputs"],
                "GPU full source input differs from CPU preflight")
        attentions=[layer.self_attn for layer in model.model.language_model.layers]
        require(len(attentions)==LAYERS and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual 28-layer route changed")

        def before_model(_module,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=14 and
                    active.get("name")==NAMES[counts["model_forwards"]-1],
                    "unexpected model call/order")
            car.verify_actual_input(kwargs,active["input"],media=active["media"])
            active["top_input"]={k:car.base.tensor_hash(kwargs[k]) for k in
                                 ("input_ids","attention_mask","position_ids","cache_position")}
        def before_vision(_module,_args):
            counts["vision_forwards"]+=1
            require(active.get("media") and counts["vision_forwards"]<=4,
                    "unexpected vision call")
        def on_rotary(_module,args,output):
            require(len(args)>=2 and torch.equal(args[1],active["input"]["position_ids"]),
                    "actual rotary positions changed")
            cos,sin=output;n=active["input"]["input_ids"].shape[1]
            require(cos.shape==sin.shape==(4,n,DIM) and not active["rotary_values"],
                    "actual rotary shape/count changed")
            active["rotary_values"].append((cos,sin))
            active["rotary_hashes"]={"cos":car.base.tensor_hash(cos),
                                     "sin":car.base.tensor_hash(sin)}
        handles.extend((model.register_forward_pre_hook(before_model,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision),
                        model.model.language_model.rotary_emb.register_forward_hook(on_rotary)))
        for i,attention in enumerate(attentions):
            def at_entry(_module,_args,kwargs,layer=i):
                mask=kwargs.get("attention_mask")
                car.verify_attention(mask,active["expected_mask"])
                embed=kwargs.get("position_embeddings")
                require(len(active["rotary_values"])==1 and isinstance(embed,tuple) and
                        len(embed)==2 and
                        torch.equal(embed[0],active["rotary_values"][0][0]) and
                        torch.equal(embed[1],active["rotary_values"][0][1]),
                        "actual attention rotary consumer changed")
                record={"layer":layer,"mask_sha256":car.base.tensor_hash(mask),
                        "rotary":active["rotary_hashes"]}
                if active["kind"]=="suffix":
                    cache=active["cache"]
                    require(kwargs.get("past_key_values") is cache and
                            cache.get_seq_length(layer)==WIDTH and
                            torch.equal(kwargs.get("cache_position"),active["input"]["cache_position"]),
                            "actual suffix cache slots changed")
                    record["segments"]=older_layer_selected(cache.layers[layer],
                        active["base_segments"][layer],
                        active["donor_blocks"]["layers"][layer],
                        base_origin=active["base_origin"],
                        donor_origin=active["donor_origin"],mode=active["mode"])
                active["attn"].append(record)
            handles.append(attention.register_forward_pre_hook(at_entry,with_kwargs=True))
        for i,layer in enumerate(model.model.language_model.layers):
            def at_output(_module,_args,idx=i):
                if active["kind"]!="suffix":return
                cache=active["cache"]
                require(cache.get_seq_length(idx)==END and
                        {name:car.base.tensor_hash(getattr(cache.layers[idx],name)[:,:,:WIDTH,:])
                         for name in ("keys","values")}==active["patched_digest"][idx],
                        "historical K/V changed during suffix")
                row={name:car.base.tensor_hash(getattr(cache.layers[idx],name)[[0,1,3],:,WIDTH:END,:])
                     for name in ("keys","values")}
                active["after"].append({"layer":idx,"companion_suffix":row,
                                        "historical_digest":active["patched_digest"][idx]})
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(at_output))

        def invoke(name,inputs,*,media,cache=None,base_origin=None,donor_origin=None,
                   donor_blocks=None,base_segments=None,original_digest=None):
            i=NAMES.index(name)
            kind="suffix" if i>=4 else "prefill" if i>=2 else "full"
            mode=cell_mode(name) if kind=="suffix" else kind
            expected_mask=(car.expected_suffix_mask(fulls[base_origin]) if kind=="suffix" else
                           car.base.native_4d(inputs["attention_mask"]))
            active.clear();active.update(name=name,kind=kind,mode=mode,input=inputs,
                                         media=media,cache=cache,base_origin=base_origin,
                                         donor_origin=donor_origin,donor_blocks=donor_blocks,
                                         base_segments=base_segments,expected_mask=expected_mask,
                                         attn=[],after=[],rotary_values=[],rotary_hashes={})
            if kind=="suffix":
                with scope(cache,original_digest,mode=mode,base_origin=base_origin,
                           donor_origin=donor_origin,donor_blocks=donor_blocks):
                    active["patched_digest"]=cache_digest(cache)
                    for layer in range(LAYERS):
                        older_layer_selected(cache.layers[layer],base_segments[layer],
                            donor_blocks["layers"][layer],base_origin=base_origin,
                            donor_origin=donor_origin,mode=mode)
                    with torch.inference_mode():logits=model(**inputs).logits[:,-1,:].detach().float().cpu()
            else:
                with torch.inference_mode():logits=model(**inputs).logits[:,-1,:].detach().float().cpu()
            torch.cuda.synchronize(device)
            require(logits.shape==(4,152670) and torch.isfinite(logits).all().item() and
                    [x["layer"] for x in active["attn"]]==list(range(LAYERS)) and
                    len(active["rotary_values"])==1 and
                    (kind!="suffix" or [x["layer"] for x in active["after"]]==list(range(LAYERS))),
                    f"{name} actual consumer/vector incomplete")
            directory=OUT/"cells"/f"{i+1:02d}-{name}";directory.mkdir(parents=True,exist_ok=False)
            torch.save(logits,directory/"full-batch-vocabulary.pt")
            write_new(directory/"input.json",{k:v.detach().cpu().tolist() for k,v in inputs.items()
                if k in ("input_ids","attention_mask","position_ids","cache_position")})
            consumer={"name":name,"kind":kind,"mode":mode,
                      "actual_input":active["top_input"],"rotary":active["rotary_hashes"],
                      "attention":active["attn"],"after_suffix":active["after"],
                      "patched_digest":active.get("patched_digest"),
                      "restored_digest":cache_digest(cache) if kind=="suffix" else None,
                      "expected_mask_sha256":car.base.tensor_hash(expected_mask)}
            write_new(directory/"consumer.json",consumer)
            cell={"call":i+1,"name":name,"origins":axes(a["cells"][i]),
                  "input":bind(directory/"input.json"),
                  "vector":bind(directory/"full-batch-vocabulary.pt"),
                  "consumer":bind(directory/"consumer.json"),
                  "counts_after":dict(counts)}
            receipt["cells"].append(cell)
            return logits,consumer

        full_vectors={}
        for name,origin in (("full_AF","AF"),("full_FF","FF")):
            logits,_=invoke(name,fulls[origin],media=True)
            ref=torch.load(a["saved_full_vector_references"]["native_AF" if origin=="AF" else "FF"]["path"],
                           map_location="cpu",weights_only=True)["logits"]
            err=float((logits-ref).abs().max())
            require(err<=TOL,f"{name} accepted all-four reference mismatch: {err}")
            indexes=range(4) if origin=="AF" else (0,1,3)
            parity=[car.base._trace_compare(logits=logits[j],trace=trace,batch_index=j,
                        absolute_offset=22,token_id=raw[j]["token_ids"][22],
                        role="original_AF_x1" if origin=="AF" and j==TARGET else "source_companion_x1",
                        atol=TOL) for j in indexes]
            require(all(x["passed"] for x in parity),f"{name} source trace parity failed")
            receipt["cells"][-1].update(saved_reference_max_abs=err,source_trace_parity=parity)
            full_vectors[origin]=logits

        caches={};segments={};blocks={};digests={}
        for name,origin in (("prefill_AF","AF"),("prefill_FF","FF")):
            cache=DynamicCache();inputs=car.split_inputs(fulls[origin],cache,"prefill")
            invoke(name,inputs,media=True,cache=cache)
            require(cache.get_seq_length()==WIDTH,"prefill cache width changed")
            caches[origin]=cache;segments[origin]=car.segments(cache)
            blocks[origin]=car.blocks(cache,origin);digests[origin]=cache_digest(cache)
            receipt["cells"][-1]["cache_digest"]=digests[origin]
        require(all(segments["AF"][i][axis][span]==segments["FF"][i][axis][span]
                    for i in range(LAYERS) for axis in ("keys","values")
                    for span in ("prompt","companions")),
                "prefill target prehistory or companion K/V changed")
        torch.save(blocks,OUT/"historical-blocks.pt")
        write_new(OUT/"prefill-origins.json",{"segments":segments,"digests":digests,
                  "blocks":bind(OUT/"historical-blocks.pt")})

        suffix_vectors={};suffix_consumers={}
        for name in NAMES[4:]:
            i=NAMES.index(name);cell=a["cells"][i]
            base_origin=cell["base_origin"]
            donor_origin=(cell["older_K_origin"] if cell["kind"] in ("joint","K") else
                          cell["older_V_origin"] if cell["kind"]=="V" else base_origin)
            require(len(receipt["cells"])==i and
                    (i<10 or all(x in suffix_vectors for x in NAMES[4:10])),
                    "out-of-order or premature component call")
            inputs=car.split_inputs(fulls[base_origin],caches[base_origin],"suffix")
            logits,consumer=invoke(name,inputs,media=False,cache=caches[base_origin],
                                   base_origin=base_origin,donor_origin=donor_origin,
                                   donor_blocks=blocks[donor_origin],
                                   base_segments=segments[base_origin],
                                   original_digest=digests[base_origin])
            require(consumer["restored_digest"]==digests[base_origin],
                    f"{name} cache restoration changed")
            if cell["kind"]=="anchor":
                err=max(float((logits-full_vectors[base_origin]).abs().max()),
                        float((logits-torch.load(a["anchor_references"][base_origin]["path"],
                                        map_location="cpu",weights_only=True)).abs().max()))
                require(err<=TOL,f"{name} fresh/saved anchor mismatch: {err}")
                receipt["cells"][-1]["full_and_saved_anchor_max_abs"]=err
            else:
                anchor=suffix_vectors["anchor_"+base_origin]
                companion_err=max(float((logits[j]-anchor[j]).abs().max()) for j in (0,1,3))
                require(companion_err<=TOL and
                        [x["companion_suffix"] for x in consumer["after_suffix"]]==
                        [x["companion_suffix"] for x in
                         suffix_consumers["anchor_"+base_origin]["after_suffix"]],
                        f"{name} companion suffix K/V or logits changed")
                receipt["cells"][-1]["companion_anchor_max_abs"]=companion_err
                if cell["kind"]=="sham":
                    err=float((logits-anchor).abs().max())
                    require(err<=TOL,f"{name} own older-write sham mismatch: {err}")
                    receipt["cells"][-1]["sham_anchor_max_abs"]=err
                if cell["kind"]=="joint":
                    reference=a["joint_references"][name]
                    prior_vec=torch.load(reference["vector"]["path"],map_location="cpu",weights_only=True)
                    err=float((logits-prior_vec).abs().max())
                    require(err<=TOL and
                            json.loads(Path(receipt["cells"][-1]["input"]["path"]).read_text())==
                            json.loads(Path(reference["input"]["path"]).read_text()),
                            f"{name} complementary accepted hybrid mismatch: {err}")
                    receipt["cells"][-1]["complementary_joint_max_abs"]=err
            suffix_vectors[name]=logits;suffix_consumers[name]=consumer
        require(counts=={"model_forwards":14,"vision_forwards":4,"generated_tokens":0},
                "finite 14/4/0 counts changed")
        receipt["effective_identity"]=identity
        receipt["status"]="candidate_raw_complete"
    except Exception as exc:
        receipt["status"]="technical_failure"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),
                            "traceback":traceback.format_exc()}
    finally:
        for handle in handles:handle.remove()
        receipt["internal_seconds"]=time.monotonic()-started
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch.cuda.is_available():
            receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
            receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device)
        receipt["artifact_bytes_before_receipt"]=sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"]=os.getpid()
        write_new(OUT/"receipt.json",receipt)
        print(json.dumps({"status":receipt["status"],"counts":counts,
                          "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_raw_complete","technical failure; no retry")


def run_parent():
    checked();require(not (OUT/"launch.json").exists() and not (OUT/"outer.json").exists(),
                      "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    command=["python","-B","-m","probes.training_set_completion.recurrence_older_kv_split.run","gpu"]
    began=time.monotonic();unix=time.time()
    with (OUT/"stdout.log").open("x") as stdout,(OUT/"stderr.log").open("x") as stderr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        code=child.wait()
    outer={"command":command,"started_unix":unix,"outer_seconds":time.monotonic()-began,
           "child_pid":child.pid,"returncode":code,"terminal":True,
           "stdout":bind(OUT/"stdout.log"),"stderr":bind(OUT/"stderr.log")}
    write_new(OUT/"outer.json",outer)
    print(json.dumps(outer))
    require(code==0,"terminal child failure; no retry")


def reduce_vectors(vectors):
    probabilities={name:torch.softmax(vec[TARGET].double(),dim=-1)
                   for name,vec in vectors.items() if not name.startswith("prefill")}
    def tv(x,y):return float((probabilities[x]-probabilities[y]).abs().sum()/2)
    bases={};guard=1e-6
    for origin,donor in (("AF","FF"),("FF","AF")):
        anchor="anchor_"+origin;joint=f"joint_{origin}_from_{donor}"
        v=f"V_{origin}_from_{donor}";k=f"K_{origin}_from_{donor}"
        distance=tv(anchor,joint)
        v_joint=tv(v,joint);k_joint=tv(k,joint)
        rv=v_joint/distance if distance>0 else None
        rk=k_joint/distance if distance>0 else None
        if distance<=1e-6:category="negligible_separation"
        elif min(abs(distance-1e-6),abs(rv-.5),abs(rk-.5),
                 abs((rk-rv)-.1),abs((rv-rk)-.1))<=guard:category="numerical_HOLD"
        elif rv<.5 and rk-rv>.1:category="selective_V"
        elif rk<.5 and rv-rk>.1:category="selective_K"
        elif rv<.5 and rk<.5:category="both_below_half"
        else:category="neither"
        bases[origin]={"joint":joint,"V":v,"K":k,"D_native_to_joint":distance,
                       "TV_V_to_joint":v_joint,"TV_K_to_joint":k_joint,
                       "TV_V_to_native":tv(v,anchor),"TV_K_to_native":tv(k,anchor),
                       "r_V":rv,"r_K":rk,"V_below_half":None if rv is None else rv<.5,
                       "K_below_half":None if rk is None else rk<.5,"category":category}
    categories=[bases[o]["category"] for o in ("AF","FF")]
    if categories==["selective_V"]*2:shared="primary_V_selective_pass"
    elif categories==["selective_K"]*2:shared="K_comparator_pass"
    elif "numerical_HOLD" in categories:shared="numerical_HOLD"
    elif "negligible_separation" in categories:shared="negligible_separation"
    else:shared="mixed_or_neither"
    secondary={}
    for name,p in probabilities.items():
        z=vectors[name][TARGET].double();top=torch.topk(z,2)
        secondary[name]={"winner":int(top.indices[0]),"runner":int(top.indices[1]),
                         "gap":float(top.values[0]-top.values[1]),
                         "z_151671_minus_151670":float(z[151671]-z[151670]),
                         "fixed_tokens":{str(j):{"probability":float(p[j]),
                                                  "rank":int((z>z[j]).sum())+1}
                                         for j in (151670,151671)}}
    return {"bases":bases,"shared_outcome":shared,"secondary":secondary}


def readback():
    a,p=checked()
    require(not (OUT/"readback.json").exists() and not (UNIT/"candidate-results.md").exists(),
            "candidate readback already exists")
    r=json.loads((OUT/"receipt.json").read_text());outer=json.loads((OUT/"outer.json").read_text())
    require(r["status"]=="candidate_raw_complete" and r["admission"]==bind(ADMISSION) and
            r["preflight"]==bind(PREFLIGHT) and r["producer"]==bind(Path(__file__)) and
            r["counts"]=={"model_forwards":14,"vision_forwards":4,"generated_tokens":0} and
            outer["returncode"]==0 and outer["terminal"] and
            outer["child_pid"]==r["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists(),
            "terminal source/count/receipt changed")
    record_guard(r["cells"],a["cells"])
    q=car.base.load_qwen_components_from_options(car.base.QwenLoadOptions(
        base_model=str(car.base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold reader loaded language model")
    batch,raw,trace,sr,_=car.prior.source(q,a,torch.device("cpu"))
    require(sr["input_identity"]==p["input_identity"] and
            int(q.tokenizer.pad_token_id)==p["pad_id"],"cold source changed")
    fulls={o:car.make_full(car.base.ConfigOnlyRope(),batch,raw,p["pad_id"],o)
           for o in ("AF","FF")}
    pre=json.loads((OUT/"prefill-origins.json").read_text())
    require(pre["blocks"]==bind(OUT/"historical-blocks.pt") and
            set(pre["segments"])==set(pre["digests"])=={"AF","FF"},
            "cold prefill origin/capture changed")
    snapshots=torch.load(pre["blocks"]["path"],map_location="cpu",weights_only=True)
    require(set(snapshots)=={"AF","FF"},"cold donor set changed")
    for origin in ("AF","FF"):
        require(snapshots[origin]["origin"]==origin and
                len(snapshots[origin]["layers"])==len(pre["segments"][origin])==
                len(pre["digests"][origin])==LAYERS,"cold donor layers changed")
        for i,row in enumerate(snapshots[origin]["layers"]):
            for span in ("older","latest"):
                for name in ("keys","values"):
                    tensor=row[span][name]
                    require(tensor.shape==(HEADS,9,DIM) and torch.isfinite(tensor).all().item() and
                            car.base.tensor_hash(tensor)==pre["segments"][origin][i][name][span],
                            "cold donor block shape/hash changed")
    require(all(pre["segments"]["AF"][i][name][span]==
                pre["segments"]["FF"][i][name][span]
                for i in range(LAYERS) for name in ("keys","values")
                for span in ("prompt","companions")),
            "cold pre-object/companion K/V changed")
    vectors={};consumers={};records=[]
    for i,cell in enumerate(r["cells"]):
        name=cell["name"];admitted=a["cells"][i];base_origin=admitted["base_origin"]
        for key in ("input","vector","consumer"):
            require(bind(cell[key]["path"])==cell[key],f"cold {name} {key} binding changed")
        saved=json.loads(Path(cell["input"]["path"]).read_text())
        consumer=json.loads(Path(cell["consumer"]["path"]).read_text())
        vector=torch.load(cell["vector"]["path"],map_location="cpu",weights_only=True)
        require(consumer["name"]==name and vector.shape==(4,152670) and
                torch.isfinite(vector).all().item() and
                [x["layer"] for x in consumer["attention"]]==list(range(LAYERS)) and
                len(consumer["after_suffix"])==(LAYERS if i>=4 else 0),
                "cold vector/consumer layer evidence changed")
        full=fulls[base_origin]
        expected=full if i<2 else car.split_inputs(full,object(),"prefill" if i<4 else "suffix")
        for key in ("input_ids","attention_mask","position_ids","cache_position"):
            require(saved[key]==expected[key].cpu().tolist() and
                    consumer["actual_input"][key]==car.base.tensor_hash(expected[key]),
                    f"cold {name} actual source/{key} changed")
        mask=car.expected_suffix_mask(full) if i>=4 else car.base.native_4d(expected["attention_mask"])
        require(consumer["expected_mask_sha256"]==car.base.tensor_hash(mask) and
                all(row["mask_sha256"]==car.base.tensor_hash(mask) for row in consumer["attention"]),
                f"cold {name} actual all-layer mask changed")
        if i>=4:
            mode=cell_mode(name)
            require(consumer["mode"]==mode and
                    consumer["restored_digest"]==pre["digests"][base_origin] and
                    len(consumer["patched_digest"])==LAYERS,
                    f"cold {name} mode/restoration changed")
            for layer,row in enumerate(consumer["attention"]):
                for axis,origin_key in (("keys","older_K_origin"),("values","older_V_origin")):
                    got=row["segments"][axis]
                    base_row=pre["segments"][base_origin][layer][axis]
                    donor=admitted[origin_key]
                    wanted=car.base.tensor_hash(snapshots[donor]["layers"][layer]["older"][axis])
                    require(all(got[span]==base_row[span] for span in
                                ("prompt","latest","companions")) and
                            got["older"]==wanted and
                            consumer["patched_digest"][layer][axis]==
                                consumer["after_suffix"][layer]["historical_digest"][axis],
                            f"cold {name} layer {layer} older {axis}/complement changed")
        else:
            require(consumer["patched_digest"] is None and
                    consumer["restored_digest"] is None,
                    "cold full/prefill cache receipt changed")
        vectors[name]=vector;consumers[name]=consumer
        records.append({"call":i+1,"name":name,"origins":axes(admitted),
                        "vector":cell["vector"],"input":cell["input"],
                        "consumer":cell["consumer"]})
    for name,origin in (("full_AF","AF"),("full_FF","FF")):
        reference=torch.load(a["saved_full_vector_references"]["native_AF" if origin=="AF" else "FF"]["path"],
                             map_location="cpu",weights_only=True)["logits"]
        require(float((vectors[name]-reference).abs().max())<=TOL,
                "cold accepted full reference changed")
        indexes=range(4) if origin=="AF" else (0,1,3)
        require(all(car.base._trace_compare(logits=vectors[name][j],trace=trace,batch_index=j,
                    absolute_offset=22,token_id=raw[j]["token_ids"][22],role="cold_AF_or_companion",
                    atol=TOL)["passed"] for j in indexes),"cold source trace changed")
    for origin in ("AF","FF"):
        accepted=torch.load(a["anchor_references"][origin]["path"],map_location="cpu",weights_only=True)
        require(float((vectors["anchor_"+origin]-vectors["full_"+origin]).abs().max())<=TOL and
                float((vectors["anchor_"+origin]-accepted).abs().max())<=TOL and
                float((vectors["sham_"+origin]-vectors["anchor_"+origin]).abs().max())<=TOL,
                "cold full/anchor/sham all-four parity failed")
    for name in NAMES[8:]:
        admitted=a["cells"][NAMES.index(name)];origin=admitted["base_origin"]
        require(max(float((vectors[name][j]-vectors["anchor_"+origin][j]).abs().max())
                    for j in (0,1,3))<=TOL and
                [x["companion_suffix"] for x in consumers[name]["after_suffix"]]==
                [x["companion_suffix"] for x in consumers["anchor_"+origin]["after_suffix"]],
                "cold companion K/V or logits changed")
    for name in NAMES[8:10]:
        ref=a["joint_references"][name]
        accepted=torch.load(ref["vector"]["path"],map_location="cpu",weights_only=True)
        require(float((vectors[name]-accepted).abs().max())<=TOL and
                json.loads(Path(records[NAMES.index(name)]["input"]["path"]).read_text())==
                json.loads(Path(ref["input"]["path"]).read_text()),
                "cold joint complementary hybrid qualification failed")
    reduced=reduce_vectors(vectors)
    result={"status":"candidate_cold_readback_passed","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),
            "receipt":bind(OUT/"receipt.json"),"outer":bind(OUT/"outer.json"),
            "prefill_origins":bind(OUT/"prefill-origins.json"),
            "cells":records,"counts":r["counts"],"outcomes":reduced,
            "model_loads":0,"model_forwards":0,"vision_forwards":0,
            "cuda_calls":0,"gpu_seconds_added_by_readback":0}
    write_new(OUT/"readback.json",result)
    lines=["# Older-record K/V split candidate", "",
           "Status: **candidate cold-readback passed; lead acceptance pending.**", "",
           f"Protocol SHA `{SHAS[PROTOCOL]}`; admission SHA `{SHAS[ADMISSION]}`; "
           f"producer SHA `{bind(Path(__file__))['sha256']}`.",
           f"Raw receipt SHA `{result['receipt']['sha256']}`; cold readback SHA "
           f"`{bind(OUT/'readback.json')['sha256']}`.", "",
           "## Frozen full-vocabulary endpoint", "",
           f"Shared outcome **{reduced['shared_outcome']}**. FP64 full152670-way softmax; "
           "both AF and FF bases retained.", "",
           "| Base | D native→joint | TV V→joint | TV K→joint | r_V | r_K | Category |",
           "|---|---:|---:|---:|---:|---:|---|" ]
    for origin in ("AF","FF"):
        x=reduced["bases"][origin]
        lines.append(f"| {origin} | {x['D_native_to_joint']:.12g} | {x['TV_V_to_joint']:.12g} | "
                     f"{x['TV_K_to_joint']:.12g} | {x['r_V']:.12g} | {x['r_K']:.12g} | {x['category']} |")
    lines += ["", "The header `[151646,8987,151647,151648]` was replayed conditioning; "
              "zero tokens were generated. This fixed-cache-origin contrast does not identify "
              "a physical owner, a literal coordinate copy, or a natural mediation percentage.",
              "", "## Qualification and resources", "",
              "Calls1–10 qualified before V/K-only readouts. Fresh full references, cached anchors, "
              "independent older K+V shams and both complete older swaps passed the frozen all-four "
              "2e-4 gates, including complementary accepted hybrid references. Actual all28-layer "
              "older K/V donor, untouched latest/prehistory/companion cache, native mask/positions, "
              "companion suffix, and finally restoration were checked and cold-read separately.",
              "",f"Actual calls {r['counts']['model_forwards']} model / {r['counts']['vision_forwards']} vision / "
              f"{r['counts']['generated_tokens']} generated. Parent outer {outer['outer_seconds']:.9f} s; "
              f"internal {r['internal_seconds']:.9f} s. Prior sequence 0.560139886111397 GPUh; "
              f"new cumulative {0.560139886111397+outer['outer_seconds']/3600:.12f} GPUh.",
              f"Peak RSS {r['rss_peak_kib']} KiB; GPU allocated/reserved "
              f"{r['gpu_peak_allocated_bytes']}/{r['gpu_peak_reserved_bytes']} bytes; "
              f"artifact bytes before receipt {r['artifact_bytes_before_receipt']}. "
              f"Terminal child PID {outer['child_pid']}, exit {outer['returncode']}.",
              "", "All individual raw vectors, source inputs, consumer hashes, component-to-native "
              "distances and fixed-token probabilities/ranks are indexed in the cold "
              "[readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-kv-split/attempt-001/readback.json).",
              "", "F physical owner remains HOLD. This is not self-accepted and admits no successor.", ""]
    (UNIT/"candidate-results.md").write_text("\n".join(lines))
    print(json.dumps({"status":result["status"],"shared":reduced["shared_outcome"],
                      "outer_seconds":outer["outer_seconds"]}))


def selfcheck():
    a=contract()
    q=car.base.load_qwen_components_from_options(car.base.QwenLoadOptions(
        base_model=str(car.base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"selfcheck loaded model")
    batch,raw,_,_,_=car.prior.source(q,a,torch.device("cpu"))
    checks,_=cpu_checks(q,batch,raw,int(q.tokenizer.pad_token_id),a)
    print(json.dumps({"status":"cpu_selfcheck_passed","checks":checks}))


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("action",choices=("selfcheck","preflight","run","gpu","readback"))
    {"selfcheck":selfcheck,"preflight":preflight,"run":run_parent,
     "gpu":gpu_child,"readback":readback}[arg.parse_args().action]()


if __name__=="__main__":main()
