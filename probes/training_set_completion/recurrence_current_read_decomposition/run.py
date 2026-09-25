"""One fixed current-query attention decomposition, full prefix, six calls."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as qmod
from transformers.integrations import sdpa_attention as sdpa

from probes.training_set_completion.recurrence_coordinate_query_routing import run as prior


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-current-read-decomposition"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
ORIGINAL_PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
PREFLIGHT = UNIT / "supporting/attempt-002-preflight.json"
REPAIR = UNIT / "lead-repair-attempt-002.json"
FIXTURE = UNIT / "supporting/attempt-002-device-qualification.json"
ORIGINAL_OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-current-read-decomposition/attempt-001")
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-current-read-decomposition/attempt-002")
SHAS = {PROTOCOL:"9bc6f37fa40b09948f68aa36c553c166cf92e73c4ac4d83478579e955a69c905",
        ADMISSION:"9673d1d1b5300e4c16d256b6b68ad5ea143ac82bb353159f43de6a3fae5ea312",
        REPAIR:"28cde4d38811513b18e3859c33e2b5b273754081acd8c85a118ca188b0b3087d"}
ARMS = ("native_anchor","current_mask_anchor","identity_reconstruction",
        "full_mask_bridge","remove_H","redistribute_R")
TARGET,C,WIDTH,H0,H1,LAYERS,QH,KVH,DIM,TOL,HEAD_REL = 2,1376,1377,1362,1371,28,16,8,128,2e-4,5e-5
KEYS = prior.KEYS
base = prior.base
require,bind,write_new = base.require,base.bind,base.write_new


def contract():
    for path,sha in SHAS.items():require(bind(path)["sha256"]==sha,f"contract changed: {path}")
    a=json.loads(ADMISSION.read_text())
    require(a["status"]=="lead-admitted-six-call-current-read-decomposition" and
            a["worker_thread"]=="01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"]=="gpt-6-sol" and a["worker_effort"]=="xhigh" and
            [x["name"] for x in a["cells"]]==list(ARMS) and
            [x["write_formula"] for x in a["cells"]]==[None,None,"H+R","O_R","R","H+O_R"] and
            [x["mask_mode"] for x in a["cells"]]==["native","current_only"]+["native"]*4 and
            a["qualification"]["headout_relative_bound"]==HEAD_REL and
            a["qualification"]["full_vocabulary_max_abs_error"]==TOL and
            a["counts"]["model_forwards"]==a["counts"]["vision_forwards"]==6 and
            a["counts"]["generated_tokens"]==a["counts"]["reused_calls"]==
                a["counts"]["nonmodel_CUDA_diagnostics"]==0 and
            a["planning"]["artifact_envelope_bytes"]==3*1024**3 and
            a["owned_paths"]["raw_root"]==str(ORIGINAL_OUT),"admitted cells/gates/cost changed")
    ruling=json.loads(REPAIR.read_text())
    require(ruling["status"]=="lead-authorized-bounded-serialization-repair-and-conditional-attempt002" and
            ruling["worker_thread"]=="01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            ruling["worker_model"]=="gpt-6-sol" and ruling["worker_effort"]=="xhigh" and
            ruling["fresh_paths"]["raw_root"]==str(OUT) and
            ruling["fresh_paths"]["preflight"]==str(PREFLIGHT) and
            ruling["fresh_paths"]["fixture_receipt"]==str(FIXTURE) and
            ruling["counts"]["attempt002_model"]==ruling["counts"]["attempt002_vision"]==6 and
            ruling["counts"]["maximum_additional_nonmodel_CUDA_fixture_invocations"]==1 and
            ruling["accounting"]["charged_after_attempt001_gpu_hours"]==
                a["prior_sequence_gpu_hours"]+ruling["accounting"]["failed_outer_seconds"]/3600,
            "attempt002 ruling/scope/charge changed")
    for key in ("protocol","admission","failed_candidate","failure_packet",
                "lead_actual_cold_RED_and_numeric_audit","failed_preflight",
                "immutable_source_capture","failed_receipt","failed_outer",
                "failed_raw_native","failed_inputs"):
        require(bind(ruling[key]["path"])==ruling[key],f"failed evidence changed: {key}")
    require(ruling["immutable_source_capture"]["sha256"]==
            ruling["failed_producer"]["sha256"],"failed producer capture differs")
    for name in ("protocol","cpu_feasibility","cpu_bindings","cpu_acceptance",
                 "predecessor_acceptance","predecessor_result","predecessor_admission",
                 "predecessor_receipt","predecessor_readback","predecessor_outer",
                 "predecessor_preflight","predecessor_verification","earlier_mass_counterexample",
                 "source_panel"):
        require(bind(a[name]["path"])==a[name],f"binding changed: {name}")
    for name in ("raw","trace","runtime_receipt","image"):
        require(bind(a["source_bindings"][name]["path"])==a["source_bindings"][name],
                f"source changed: {name}")
    for name,item in a["maintained_route"].items():
        require(bind(item["path"])==item,f"maintained route changed: {name}")
    cross=a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"])==cross["maintained"] and
            cross["maintained"]["sha256"]==cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"])==cross["prior_acceptance"],
            "loader crosswalk changed")
    old_a,_=prior.contract()
    require(a["source_bindings"]["raw"]==old_a["source_bindings"]["raw"] and
            a["source_bindings"]["trace"]==old_a["source_bindings"]["trace"] and
            a["conditioning"]["current_prefix"]==old_a["conditioning"]["current_prefix"] and
            a["model_geometry"]["query_heads"]==QH and
            a["model_geometry"]["kv_heads"]==KVH and
            a["model_geometry"]["head_dim"]==DIM and
            a["model_geometry"]["target_query_physical"]==C and
            a["model_geometry"]["history_keys_physical"]==[H0,H1],
            "source/current geometry changed")
    for name in ("native","current_only"):
        for kind in ("raw","inputs"):
            item=a["references"][name][kind]
            require(bind(item["path"])==item,f"{name} reference changed")
    return a,old_a


def spec(a,name):
    require(name in ARMS,"unknown cell")
    return a["cells"][ARMS.index(name)]


def full_input(model,batch,raw,pad,a):
    return prior.full_input(model,batch,raw,pad,a)


def mask_for(full,old_a,name):
    return prior.mask_for(full,old_a,"current_only" if name=="current_mask_anchor" else "native")


def nonselected_hash(tensor):
    require(tensor.shape==(4,WIDTH,QH*DIM),"o_proj input complement shape changed")
    h=hashlib.sha256()
    for part in (tensor[:TARGET],tensor[TARGET,:C],tensor[TARGET,C+1:],tensor[TARGET+1:]):
        h.update(part.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def components(q,k,v,cos,sin,native_row,*,target=TARGET,query=C,
               history=(H0,H1),group=2,formula="H+R"):
    require(target==TARGET and query==C and history==(H0,H1) and group==QH//KVH and
            formula in ("H+R","O_R","R","H+O_R") and
            q.shape==(QH,DIM) and k.shape==v.shape==(WIDTH,KVH,DIM) and
            cos.shape==sin.shape==(WIDTH,DIM) and native_row.shape==(WIDTH,) and
            native_row.dtype==torch.bool and
            all(x.dtype==torch.float32 for x in (q,k,v,cos,sin)) and
            bool(native_row[:C+1].all()) and not bool(native_row[C+1:].any()),
            "selected query/history/GQA/mask/RoPE/QKV changed")
    require(all(torch.isfinite(x).all().item() for x in (q,k,v,cos,sin)),
            "nonfinite incoming Q/K/V or rotary")
    # Match the model's FP32 RoPE operations before FP64 scores and reductions.
    qrot=q*cos[C]+qmod.rotate_half(q)*sin[C]
    krot=k*cos[:,None,:]+qmod.rotate_half(k)*sin[:,None,:]
    kr=krot.permute(1,0,2).repeat_interleave(group,dim=0)
    vr=v.permute(1,0,2).repeat_interleave(group,dim=0).double()
    scores=torch.einsum("hd,hkd->hk",qrot.double(),kr.double())*(DIM**-.5)
    history_mask=torch.zeros(WIDTH,dtype=torch.bool,device=q.device)
    history_mask[H0:H1]=True
    hmask=native_row&history_mask;rmask=native_row&~history_mask
    require(int(hmask.sum())==9 and int(rmask.sum())==1368 and
            torch.isfinite(scores[:,native_row]).all().item(),
            "history/remaining readable score partition changed")
    sh=scores[:,hmask];sr=scores[:,rmask]
    lh=torch.logsumexp(sh,dim=-1);lr=torch.logsumexp(sr,dim=-1)
    z=torch.logaddexp(lh,lr);mr=torch.exp(lr-z);mh=torch.exp(lh-z)
    require(all(torch.isfinite(x).all().item() for x in (lh,lr,z,mr,mh)) and
            bool((mr>0).all()),"remaining mass nonfinite or unrepresentable")
    h=mh[:,None]*torch.einsum("hk,hkd->hd",torch.softmax(sh,dim=-1),vr[:,hmask])
    other=torch.einsum("hk,hkd->hd",torch.softmax(sr,dim=-1),vr[:,rmask])
    r=mr[:,None]*other
    vals={"H+R":h+r,"O_R":other,"R":r,"H+O_R":h+other}
    require(all(torch.isfinite(x).all().item() for x in vals.values()),
            "nonfinite decomposed head output")
    return vals,{"remaining_mass":mr,"history_mass":mh,"logZ_H":lh,"logZ_R":lr,
                 "logZ":z,"scores":scores}


@contextmanager
def hooks(model,active):
    handles=[]
    try:
        lm=model.model.language_model
        def rotary(_m,_args,output):
            require(active.get("rotary") is None and len(output)==2 and
                    all(x.shape==(4,WIDTH,DIM) and x.dtype==torch.float32 for x in output),
                    "missing/duplicate/wrong rotary capture")
            active["rotary"]=output
        handles.append(lm.rotary_emb.register_forward_hook(rotary))
        for i,layer in enumerate(lm.layers):
            for axis,module in (("Q",layer.self_attn.q_norm),
                                ("K",layer.self_attn.k_norm),
                                ("V",layer.self_attn.v_proj)):
                def capture(_m,_args,output,*,idx=i,kind=axis):
                    row=active["qkv"][idx]
                    expect={"Q":(4,WIDTH,QH,DIM),"K":(4,WIDTH,KVH,DIM),
                            "V":(4,WIDTH,KVH*DIM)}[kind]
                    require(kind not in row and output.shape==expect and
                            output.dtype==torch.float32,"missing/duplicate/wrong Q/K/V capture")
                    part=(output[TARGET,C].detach().clone() if kind=="Q" else
                          output[TARGET].detach().clone())
                    row[kind]=part if kind!="V" else part.reshape(WIDTH,KVH,DIM)
                handles.append(module.register_forward_hook(capture))
            def mask_hook(_m,_args,kwargs,*,idx=i):
                mask=kwargs.get("attention_mask")
                require(idx==len(active["masks"]) and
                        isinstance(mask,torch.Tensor) and
                        torch.equal(mask,active["expected_mask"]) and
                        kwargs.get("past_key_values") is None and
                        torch.equal(kwargs.get("cache_position"),active["full"]["cache_position"]),
                        "actual layer mask/cache/position changed")
                active["masks"].append(base.tensor_hash(mask))
                active["layers"].append(idx)
            handles.append(layer.self_attn.register_forward_pre_hook(mask_hook,with_kwargs=True))
            def actuate(_m,args,*,idx=i,attn=layer.self_attn):
                require(idx==len(active["headouts"]) and len(args)==1 and
                        args[0].shape==(4,WIDTH,QH*DIM) and
                        len(active["qkv"][idx])==3 and active.get("rotary") is not None,
                        "selected o_proj caller missing capture or order")
                original=args[0]
                data=active["qkv"][idx];cos,sin=active["rotary"]
                native_row=active["native_mask"][TARGET,0,C]
                vals,mass=components(data["Q"],data["K"],data["V"],
                                     cos[TARGET],sin[TARGET],native_row)
                before=original[TARGET,C].reshape(QH,DIM)
                oracle=vals["O_R" if active["name"]=="current_mask_anchor" else "H+R"]
                error=float((before.double()-oracle).abs().max())
                scale=max(1.0,float(before.abs().max()),float(oracle.abs().max()))
                require(error<=HEAD_REL*scale,
                        f"actual SDPA headout reconstruction failed layer {idx}: {error}>{HEAD_REL*scale}")
                formula=spec(active["admission"],active["name"])["write_formula"]
                chosen=before if formula is None else vals[formula].to(dtype=original.dtype)
                changed=original if formula is None else original.clone()
                if formula is not None:changed[TARGET,C]=chosen.reshape(QH*DIM)
                require(torch.equal(changed[TARGET,C].reshape(QH,DIM),chosen) and
                        torch.equal(changed[:TARGET],original[:TARGET]) and
                        torch.equal(changed[TARGET,:C],original[TARGET,:C]) and
                        torch.equal(changed[TARGET,C+1:],original[TARGET,C+1:]) and
                        torch.equal(changed[TARGET+1:],original[TARGET+1:]),
                        "actuator changed unselected o_proj input")
                row={"layer":idx,"pre_selected":before.detach().cpu().clone(),
                     "post_selected":chosen.detach().cpu().clone(),
                     "oracle_error":error,"oracle_scale":scale,
                     "remaining_mass":mass["remaining_mass"].detach().cpu().clone(),
                     "history_mass":mass["history_mass"].detach().cpu().clone(),
                     "pre_complement_hash":nonselected_hash(original),
                     "post_complement_hash":None,"observed_post":False}
                active["headouts"].append(row)
                return (changed,)
            def observe(_m,args,*,idx=i):
                require(idx<len(active["headouts"]) and
                        not active["headouts"][idx]["observed_post"],
                        "missing/duplicate post-actuator consumer")
                actual=args[0];row=active["headouts"][idx]
                actual_selected=actual[TARGET,C].reshape(QH,DIM)
                row["post_complement_hash"]=nonselected_hash(actual)
                require(torch.equal(actual_selected,row["post_selected"].to(actual_selected.device)) and
                        row["pre_complement_hash"]==row["post_complement_hash"],
                        "actual o_proj consumer selected/complement changed")
                row["observed_post"]=True
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(actuate))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(observe))
        yield
    finally:
        for handle in reversed(handles):handle.remove()


def verify_serialized(fixture,batch,raw,pad,a,old_a,name,entry,stored,payload):
    require(name in ARMS and entry["arm"]==payload["arm"]==name and
            entry["order"]==ARMS.index(name) and entry["step"]==6,
            "serialized arm/order changed")
    full=full_input(fixture,batch,raw,pad,a)
    mask,native,selected=mask_for(full,old_a,name)
    actual=native if name!="current_mask_anchor" else mask
    require(all(torch.equal(stored[k].detach().cpu(),full[k].detach().cpu()) for k in KEYS) and
            entry["input_hashes"]==prior.prior.input_hashes(full,mask) and
            torch.equal(payload["actual_mask"].detach().cpu(),actual.detach().cpu()) and
            entry["actual_mask_hash"]==base.tensor_hash(actual) and
            entry["actual_layers"]==list(range(LAYERS)) and
            entry["actual_layer_mask_hashes"]==[base.tensor_hash(actual)]*LAYERS and
            entry["selected_cells"]==int(selected.sum()) and
            entry["selected_native_hash"]==base.tensor_hash(native[selected]) and
            entry["selected_actual_hash"]==base.tensor_hash(actual[selected]) and
            entry["complement_native_hash"]==entry["complement_actual_hash"]==
                base.tensor_hash(native[~selected]) and
            payload["logits"].shape==(4,152670) and
            torch.isfinite(payload["logits"]).all().item() and
            len(payload["qkv"])==len(payload["headouts"])==LAYERS and
            payload["rotary_cos"].shape==payload["rotary_sin"].shape==(WIDTH,DIM) and
            payload["historical_by_layer"].shape==(LAYERS,9,2048) and
            payload["current_by_layer"].shape==(LAYERS,6,2048) and
            payload["companions_by_layer"].shape==(LAYERS,3,2048),
            "serialized source/mask/vector/state/capture changed")
    for i,(data,row) in enumerate(zip(payload["qkv"],payload["headouts"],strict=True)):
        require(set(data)=={"Q","K","V"} and row["layer"]==i and row["observed_post"] and
                row["pre_complement_hash"]==row["post_complement_hash"] and
                len(entry["headout_complement_hashes"])==LAYERS and
                entry["headout_complement_hashes"][i]==row["post_complement_hash"],
                "serialized QKV/consumer order/complement changed")
        vals,mass=components(data["Q"],data["K"],data["V"],
                             payload["rotary_cos"],payload["rotary_sin"],
                             native[TARGET,0,C].detach().cpu())
        before=row["pre_selected"]
        oracle=vals["O_R" if name=="current_mask_anchor" else "H+R"]
        error=float((before.double()-oracle).abs().max())
        scale=max(1.0,float(before.abs().max()),float(oracle.abs().max()))
        formula=spec(a,name)["write_formula"]
        expected=before if formula is None else vals[formula].float()
        saved_error=row["oracle_error"];saved_scale=row["oracle_scale"]
        saved_post=row["post_selected"]
        fp32_bound=2*torch.finfo(torch.float32).eps*max(
            1.0,float(expected.abs().max()),float(saved_post.abs().max()))
        require(all(math.isfinite(x) for x in (error,scale,saved_error,saved_scale,fp32_bound)) and
                0<=saved_error<=HEAD_REL*saved_scale and
                0<=error<=HEAD_REL*scale and
                abs(error-saved_error)<=1e-12*max(1.0,scale,saved_scale) and
                abs(scale-saved_scale)<=1e-12*max(1.0,scale,saved_scale) and
                before.shape==saved_post.shape==expected.shape==(QH,DIM) and
                before.dtype==saved_post.dtype==expected.dtype==torch.float32 and
                torch.isfinite(before).all().item() and
                torch.isfinite(saved_post).all().item() and
                (torch.equal(expected,saved_post) if formula is None else
                 float((expected-saved_post).abs().max())<=fp32_bound) and
                all(x.shape==(QH,) and x.dtype==torch.float64 and
                    torch.isfinite(x).all().item() and bool((x>=0).all()) and
                    bool((x<=1).all()) for x in
                    (mass["remaining_mass"],mass["history_mass"],
                     row["remaining_mass"],row["history_mass"])) and
                bool((mass["remaining_mass"]>0).all()) and
                bool((row["remaining_mass"]>0).all()) and
                float((mass["remaining_mass"]-row["remaining_mass"]).abs().max())<=1e-12 and
                float((mass["history_mass"]-row["history_mass"]).abs().max())<=1e-12,
                f"cold layer {i} headout/formula/mass changed")
    return full


def reference(a,name,full,logits):
    key=spec(a,name)["reference"]
    if key is None:return None
    item=a["references"][key]
    saved=json.loads(Path(item["inputs"]["path"]).read_text())
    require(all(saved[k]==full[k].detach().cpu().tolist() for k in KEYS),
            "distinct accepted reference input changed")
    old=torch.load(item["raw"]["path"],map_location="cpu",weights_only=True)["logits"]
    error=float((logits-old).abs().max())
    require(error<=TOL,"accepted full-vector anchor differs")
    return error


def state_check(name,payload,baselines):
    native=baselines.get("native_anchor")
    if native is None:return
    for key in ("historical_by_layer","companions_by_layer"):
        require(float((payload[key]-native[key]).abs().max())<=TOL,
                f"prior/companion {key} changed")
    require(float((payload["current_by_layer"][:,:5]-
                   native["current_by_layer"][:,:5]).abs().max())<=TOL and
            max(float((payload["logits"][i]-native["logits"][i]).abs().max())
                for i in (0,1,3))<=TOL,
            "header/earlier-x1/companion changed")
    key={"identity_reconstruction":"native_anchor",
         "full_mask_bridge":"current_mask_anchor"}.get(name)
    if key:
        other=baselines[key]
        require(float((payload["logits"]-other["logits"]).abs().max())<=TOL,
                "fresh identity/bridge full-vector failed")


def cpu_checks(a,old_a,batch,raw,pad):
    from transformers.masking_utils import create_causal_mask
    fixture=base.ConfigOnlyRope()
    full=full_input(fixture,batch,raw,pad,a)
    cfg=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    cfg._attn_implementation="sdpa"
    native=base.native_4d(full["attention_mask"])
    actual=create_causal_mask(cfg,torch.empty((4,WIDTH,1)),full["attention_mask"],
                              full["cache_position"],None,position_ids=full["position_ids"][0])
    require(torch.equal(actual,native),"native installed SDPA mask differs")
    checks=["installed_native_sdpa_mask"]
    for name in ARMS:
        mask,n,selected=mask_for(full,old_a,name)
        require(torch.equal(n,native) and int(selected.sum())==
                (9 if name=="current_mask_anchor" else 0),"fixed actual rectangle changed")
        checks.append("rectangle_"+name)
    for label,kwargs in (("query",{"spans":[[1375,1376]]}),
                         ("key",{"key":(1363,1371)}),("target",{"target":3})):
        try:prior.mask_for(full,old_a,"current_only",**kwargs)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError(label)
    q=torch.randn((QH,DIM),generator=torch.Generator().manual_seed(34))*.1
    k=torch.randn((WIDTH,KVH,DIM),generator=torch.Generator().manual_seed(35))*.1
    v=torch.randn((WIDTH,KVH,DIM),generator=torch.Generator().manual_seed(36))*.1
    cos=torch.ones((WIDTH,DIM));sin=torch.zeros_like(cos)
    vals,mass=components(q,k,v,cos,sin,native[TARGET,0,C])
    require(torch.allclose(vals["H+R"],vals["R"]+vals["H+O_R"]-vals["O_R"],atol=1e-12) and
            bool((mass["remaining_mass"]>0).all()),"decomposition identity failed")
    checks.append("asymmetric_FP64_partition_identity")
    rq=q.unsqueeze(0).unsqueeze(2)
    rk=k.permute(1,0,2).repeat_interleave(2,0).unsqueeze(0)
    rv=v.permute(1,0,2).repeat_interleave(2,0).unsqueeze(0)
    sd=F.scaled_dot_product_attention(rq,rk,rv,
                                      attn_mask=native[TARGET,0,C].view(1,1,1,WIDTH),
                                      dropout_p=0.0,scale=DIM**-.5)[0,:,0]
    require(float((sd-vals["H+R"]).abs().max())<HEAD_REL,
            "CPU actual SDPA headout differs from decomposition")
    checks.append("actual_CPU_SDPA_headout")
    for label,kw in (("query",{"query":C-1}),("target",{"target":1}),
                     ("history",{"history":(H0+1,H1)}),("group",{"group":1}),
                     ("formula",{"formula":"H-R"})):
        try:components(q,k,v,cos,sin,native[TARGET,0,C],**kw)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError(label)
    for label,changed in (("mask",native[TARGET,0,C].clone()),):
        changed[H0]=False
        try:components(q,k,v,cos,sin,changed)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError(label)
    # Production hooks on a one-layer CPU module exercise the real actuator and observer.
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.q_norm=torch.nn.Identity();self.k_norm=torch.nn.Identity()
            self.v_proj=torch.nn.Identity();self.o_proj=torch.nn.Identity()
        def forward(self,value,*,attention_mask,past_key_values,cache_position):
            return self.o_proj(value)
    class Layer(torch.nn.Module):
        def __init__(self):super().__init__();self.self_attn=Attention()
    class LM(torch.nn.Module):
        def __init__(self):super().__init__();self.rotary_emb=torch.nn.Identity();self.layers=torch.nn.ModuleList([Layer()])
    fake=SimpleNamespace(model=SimpleNamespace(language_model=LM()))
    fake.model.language_model.rotary_emb.forward=lambda x:x
    native_cpu=native
    active={"admission":a,"name":"remove_H","qkv":[{}],"rotary":None,
            "masks":[],"layers":[],"headouts":[],"native_mask":native_cpu,"expected_mask":native_cpu,
            "full":full}
    probe=torch.zeros((4,WIDTH,QH*DIM));probe[TARGET,C]=vals["H+R"].float().reshape(-1)
    att=fake.model.language_model.layers[0].self_attn
    def exercise():
        fake.model.language_model.rotary_emb((cos.repeat(4,1,1),sin.repeat(4,1,1)))
        xq=torch.zeros((4,WIDTH,QH,DIM));xq[TARGET,C]=q
        xk=torch.zeros((4,WIDTH,KVH,DIM));xk[TARGET]=k
        xv=torch.zeros((4,WIDTH,KVH*DIM));xv[TARGET]=v.reshape(WIDTH,-1)
        att.q_norm(xq);att.k_norm(xk);att.v_proj(xv)
        return att(probe,attention_mask=native_cpu,past_key_values=None,
                   cache_position=full["cache_position"])
    with hooks(fake,active):
        got=exercise()
    require(torch.equal(got[TARGET,C],vals["R"].float().reshape(-1)) and
            active["headouts"][0]["observed_post"] and
            not att.o_proj._forward_pre_hooks and not att._forward_pre_hooks,
            "actual CPU writer/consumer failed")
    checks.append("actual_CPU_writer_and_independent_consumer")
    active.update(qkv=[{}],rotary=None,masks=[],layers=[],headouts=[])
    try:
        with hooks(fake,active):
            exercise()
            raise RuntimeError("forced body exception")
    except RuntimeError as exc:require(str(exc)=="forced body exception","wrong forced exception")
    require(not att.o_proj._forward_pre_hooks and not att._forward_pre_hooks,
            "forced exception left hook installed")
    checks.append("forced_exception_restoration")
    from copy import deepcopy
    nmask,native,selected=mask_for(full,old_a,"remove_H")
    seed=deepcopy(active["headouts"][0]);seed["layer"]=0
    rows=[]
    for i in range(LAYERS):
        item=deepcopy(seed);item["layer"]=i;rows.append(item)
    payload={"arm":"remove_H","logits":torch.zeros((4,152670)),
             "actual_mask":native,"qkv":[{"Q":q,"K":k,"V":v} for _ in range(LAYERS)],
             "headouts":rows,"rotary_cos":cos,"rotary_sin":sin,
             "historical_by_layer":torch.zeros((LAYERS,9,2048)),
             "current_by_layer":torch.zeros((LAYERS,6,2048)),
             "companions_by_layer":torch.zeros((LAYERS,3,2048))}
    entry={"arm":"remove_H","order":ARMS.index("remove_H"),"step":6,
           "input_hashes":prior.prior.input_hashes(full,nmask),
           "actual_mask_hash":base.tensor_hash(native),
           "actual_layers":list(range(LAYERS)),
           "actual_layer_mask_hashes":[base.tensor_hash(native)]*LAYERS,
           "selected_cells":0,"selected_native_hash":base.tensor_hash(native[selected]),
           "selected_actual_hash":base.tensor_hash(native[selected]),
           "complement_native_hash":base.tensor_hash(native[~selected]),
           "complement_actual_hash":base.tensor_hash(native[~selected]),
           "headout_complement_hashes":[seed["post_complement_hash"]]*LAYERS}
    stored={key:full[key].clone() for key in KEYS}
    verify_serialized(fixture,batch,raw,pad,a,old_a,"remove_H",entry,stored,payload)
    checks.append("actual_serialized_CPU_reader")
    def reject(label,entry2=entry,stored2=stored,payload2=payload):
        try:verify_serialized(fixture,batch,raw,pad,a,old_a,"remove_H",entry2,stored2,payload2)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError("reader accepted "+label)
    bad=deepcopy(entry);bad["order"]=5;reject("next_cell_order",entry2=bad)
    bad=deepcopy(entry);bad["arm"]="redistribute_R";reject("reference_alias",entry2=bad)
    bad=deepcopy(stored);bad["input_ids"][TARGET,H0]+=1;reject("wrong_history_input",stored2=bad)
    bad=deepcopy(stored);bad["input_ids"][TARGET,C]+=1;reject("wrong_own_prefix",stored2=bad)
    bad=deepcopy(stored);bad["input_ids"][0,C]+=1;reject("wrong_companion",stored2=bad)
    bad=deepcopy(stored);bad["position_ids"][0,TARGET,C]+=1;reject("wrong_position",stored2=bad)
    bad=deepcopy(payload);bad["actual_mask"][TARGET,0,C,H0]=False
    reject("wrong_actual_mask",payload2=bad)
    bad=deepcopy(payload);bad["headouts"][0]["post_selected"]+=1
    reject("wrong_consumed_selected",payload2=bad)
    bad=deepcopy(payload);bad["headouts"][0]["post_complement_hash"]="0"*64
    reject("unrelated_consumer_output",payload2=bad)
    return checks


def preflight():
    a,old_a=contract()
    ruling=json.loads(REPAIR.read_text())
    fixture=json.loads(FIXTURE.read_text())
    require(fixture["status"]=="qualified_nonmodel_CUDA_GREEN" and
            fixture["ruling"]==bind(REPAIR) and
            fixture["producer"]==bind(Path(__file__)) and
            fixture["model_loads"]==fixture["model_forwards"]==
                fixture["vision_forwards"]==fixture["generated_tokens"]==0 and
            fixture["outer_seconds"]>0 and fixture["returncode"]==0 and
            fixture["terminal"] and not Path(f"/proc/{fixture['child_pid']}").exists(),
            "bounded CUDA fixture not qualified")
    require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),
            "fixed attempt already prepared/launched")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preparation loaded language model")
    batch,raw,trace,sr,planning=prior.prior.source(q,old_a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id)
    require(list(batch.request_ids)==a["source"]["request_ids"] and
            sr["input_identity"]==json.loads(Path(a["source_bindings"]["runtime_receipt"]["path"]).read_text())["input_identity"] and
            int(batch.inputs["pixel_values"].numel())==24502272 and
            [len(x["token_ids"]) for x in raw]==[255,37,3084,3084] and
            list(map(len,batch.prompt_token_ids))==[1336,1362,1362,1320],
            "original source/geometry differs")
    checks=cpu_checks(a,old_a,batch,raw,pad)
    full=full_input(base.ConfigOnlyRope(),batch,raw,pad,a)
    for key in ("native","current_only"):
        saved=json.loads(Path(a["references"][key]["inputs"]["path"]).read_text())
        require(all(saved[k]==full[k].tolist() for k in KEYS),
                "distinct original-path full input reference differs")
    require(full["input_ids"].shape==(4,WIDTH) and len(checks)>=30 and
            a["planning"]["planning_25pct_payload_allowance_bytes"]<
                a["planning"]["artifact_envelope_bytes"],
            "CPU caller or capacity forecast failed")
    previous=json.loads(prior.PREFLIGHT.read_text())
    sources=[Path(__file__),Path(__file__).with_name("device_fixture.py"),
             Path(prior.__file__),Path(prior.prior.__file__),
             Path(prior.prior.old.__file__),Path(base.__file__),
             Path(inspect.getfile(qmod)),Path(inspect.getfile(sdpa)),
             Path(inspect.getfile(AutoConfig))]
    sources += [Path(x["maintained"]["path"]) for x in previous["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(sources):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("external")/path.name
        saved=base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m",
             "probes.training_set_completion.recurrence_current_read_decomposition.run"]
    packet={"status":"cpu_qualified_before_gpu","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"repair":bind(REPAIR),"fixture":bind(FIXTURE),
            "failed_attempt_outer":bind(ORIGINAL_OUT/"outer.json"),
            "attempt_prior_gpu_hours":ruling["accounting"]["charged_after_attempt001_gpu_hours"]+
                                     fixture["outer_seconds"]/3600,
            "producer":bind(Path(__file__)),
            "source_identity":sr["identity"],"source_input_identity":sr["input_identity"],
            "request_ids":list(batch.request_ids),
            "prompt_lengths":list(map(len,batch.prompt_token_ids)),
            "raw_lengths":[len(x["token_ids"]) for x in raw],
            "pad_id":pad,"special_ids":sorted(q.tokenizer.all_special_ids),
            "pixel_elements":int(batch.inputs["pixel_values"].numel()),"width":WIDTH,
            "image_grids":[list(x) for x in batch.image_grids],
            "cpu_checks":checks,"direct_source_captures":captures,
            "forecast_outer_seconds":[a["planning"]["two_x_same_shape_outer_seconds"],
                                      a["planning"]["three_x_same_shape_outer_seconds"]],
            "artifact_forecast_bytes":a["planning"]["planning_25pct_payload_allowance_bytes"],
            "commands":{"preflight":command+["preflight"],"launch":command+["launch"],
                        "model":command+["run"],"cold":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"cpu_checks":len(checks),
                      "source_captures":len(captures),"width":WIDTH,
                      "pixel_elements":packet["pixel_elements"],
                      "artifact_forecast_bytes":packet["artifact_forecast_bytes"]}))


def checked():
    a,old_a=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and
            p["protocol"]==bind(PROTOCOL) and p["admission"]==bind(ADMISSION) and
            p["repair"]==bind(REPAIR) and p["fixture"]==bind(FIXTURE) and
            p["failed_attempt_outer"]==bind(ORIGINAL_OUT/"outer.json") and
            p["attempt_prior_gpu_hours"]==
                json.loads(REPAIR.read_text())["accounting"]["charged_after_attempt001_gpu_hours"]+
                json.loads(FIXTURE.read_text())["outer_seconds"]/3600 and
            p["producer"]==bind(Path(__file__)) and p["width"]==WIDTH and
            p["pixel_elements"]==24502272 and len(p["cpu_checks"])>=30 and
            p["artifact_forecast_bytes"]<a["planning"]["artifact_envelope_bytes"],
            "frozen CPU preflight/producer differs")
    for item in p["direct_source_captures"]:
        require(bind(item["maintained"]["path"])==item["maintained"] and
                bind(item["capture"]["path"])==item["capture"],
                "direct maintained source/capture changed")
    return a,old_a,p


def launch():
    checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"outer.json").exists(),
            "fixed attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-B","-m",
             "probes.training_set_completion.recurrence_current_read_decomposition.run","run"]
    begun=time.monotonic()
    with (OUT/"stdout.log").open("x") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        code=child.wait()
    packet={"command":command,"child_pid":child.pid,
            "outer_seconds":time.monotonic()-begun,"returncode":code,"terminal":True}
    write_new(OUT/"outer.json",packet)
    print(json.dumps(packet))
    require(code==0,"model attempt failed; no retry")


def run():
    a,old_a,p=checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
            "fixed attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();device=torch.device("cuda:0");handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"generated_tokens":0,"reused_calls":0}
    receipt={"status":"running","pid":os.getpid(),"begun_unix":time.time(),
             "protocol":bind(PROTOCOL),"admission":bind(ADMISSION),
             "repair":bind(REPAIR),"fixture":bind(FIXTURE),
             "failed_attempt_outer":bind(ORIGINAL_OUT/"outer.json"),
             "preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
             "counts":counts,"cells":[]}
    write_new(OUT/"launch.json",receipt)
    active={}
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q,identity=base.load_model("untied",device)
        expected=p["source_identity"]
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in expected.items() if k!="loader_source"} and
                all(identity["loader_source"][k]==expected["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"]==a["loader_crosswalk"]["maintained"]["path"],
                "effective checkpoint/loader differs")
        model=q.model.eval()
        batch,raw,trace,sr,planning=prior.prior.source(q,old_a,device)
        require(sr["input_identity"]==p["source_input_identity"] and
                int(q.tokenizer.pad_token_id)==p["pad_id"] and
                sorted(q.tokenizer.all_special_ids)==p["special_ids"],
                "GPU source/tokenizer differs")
        layers=list(model.model.language_model.layers)
        require(len(layers)==LAYERS and
                all(isinstance(x.self_attn,qmod.Qwen3VLTextAttention) for x in layers),
                "actual all-layer attention route changed")
        def top(_m,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=6,"model-forward cap")
            active["actual_input"]=prior.prior.input_hashes(kwargs,kwargs["attention_mask"])
            require(active["actual_input"]==active["expected_input_hashes"],
                    "top-level original full input changed")
        def vision(_m,_args):
            counts["vision_forwards"]+=1
            require(counts["vision_forwards"]<=6,"vision-forward cap")
        handles += [model.register_forward_pre_hook(top,with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(vision)]
        for i,layer in enumerate(layers):
            def state_hook(_m,_args,output,*,idx=i):
                value=output[0] if isinstance(output,tuple) else output
                require(isinstance(value,torch.Tensor) and value.shape==(4,WIDTH,2048) and
                        idx==len(active["history"]),"decoder state shape/order changed")
                active["history"].append(value[TARGET,H0:H1].detach().cpu().float())
                active["current"].append(value[TARGET,H1:WIDTH].detach().cpu().float())
                active["companions"].append(value[[0,1,3],-1].detach().cpu().float())
            handles.append(layer.register_forward_hook(state_hook))
        receipt["effective_identity"]=identity
        baselines={}
        with torch.inference_mode():
            for name in ARMS:
                if name=="remove_H":
                    require(len(receipt["cells"])==4 and
                            [x["arm"] for x in receipt["cells"]]==list(ARMS[:4]) and
                            counts["model_forwards"]==counts["vision_forwards"]==4,
                            "four actual references/bridge not qualified")
                full=full_input(model,batch,raw,p["pad_id"],a)
                mask,native,selected=mask_for(full,old_a,name)
                actual=native if name!="current_mask_anchor" else mask
                active.clear();active.update(admission=a,name=name,full=full,
                    expected_mask=actual,native_mask=native,selected=selected,
                    expected_input_hashes=prior.prior.input_hashes(full,mask),
                    actual_input=None,qkv=[{} for _ in range(LAYERS)],
                    rotary=None,masks=[],layers=[],headouts=[],history=[],current=[],companions=[])
                with hooks(model,active):
                    out=model(**{**full,"attention_mask":mask})
                torch.cuda.synchronize(device)
                require(active["actual_input"]==active["expected_input_hashes"] and
                        active["layers"]==list(range(LAYERS)) and
                        len(active["headouts"])==len(active["history"])==
                        len(active["current"])==len(active["companions"])==LAYERS and
                        len(active["qkv"])==LAYERS and
                        all(set(x)=={"Q","K","V"} for x in active["qkv"]) and
                        active["rotary"] is not None,
                        "actual all-layer QKV/rotary/headout/state consumers incomplete")
                logits=out.logits[:,-1,:].detach().cpu().float()
                require(logits.shape==(4,152670) and torch.isfinite(logits).all().item(),
                        "full-vocabulary output invalid")
                cos,sin=active["rotary"]
                payload={"arm":name,"logits":logits,"actual_mask":actual.detach().cpu().clone(),
                         "rotary_cos":cos[TARGET].detach().cpu().clone(),
                         "rotary_sin":sin[TARGET].detach().cpu().clone(),
                         "qkv":[{axis:value.detach().cpu().clone() for axis,value in x.items()}
                                for x in active["qkv"]],
                         "headouts":active["headouts"],
                         "historical_by_layer":torch.stack(active["history"]),
                         "current_by_layer":torch.stack(active["current"]),
                         "companions_by_layer":torch.stack(active["companions"])}
                rawpath=OUT/f"{name}.pt";torch.save(payload,rawpath)
                inputpath=OUT/f"inputs-{name}.json"
                write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                active["layer_mask_hashes"]=active["masks"]
                entry=prior.prior.entry_for(full,mask,native,selected,active,
                           int(torch.argmax(logits[TARGET]).item()),6,rawpath,inputpath)
                entry.update(arm=name,order=ARMS.index(name),
                    headout_complement_hashes=[x["post_complement_hash"] for x in active["headouts"]],
                    max_headout_error=max(x["oracle_error"] for x in active["headouts"]))
                receipt["cells"].append(entry)
                stored={k:torch.tensor(v,dtype=torch.long) for k,v in
                        json.loads(inputpath.read_text()).items()}
                verify_serialized(model,batch,raw,p["pad_id"],a,old_a,name,entry,stored,payload)
                entry["accepted_reference_max_error"]=reference(a,name,full,logits)
                state_check(name,payload,baselines)
                role="native_A" if name in ("native_anchor","identity_reconstruction") else "coordinate_A"
                entry["source_trace_parity"]=prior.prior.trace_check(logits,trace,raw,role,6)
                if name in ("native_anchor","current_mask_anchor"):baselines[name]=payload
                receipt["counts"]=dict(counts)
                write_new(OUT/f"checkpoint-{name}.json",receipt)
                del out,payload
        require(all(counts[k]==a["counts"][k] for k in counts) and
                len(receipt["cells"])==6,"fixed six-call count differs")
        receipt["status"]="candidate_raw_complete"
    except BaseException as exc:
        receipt["status"]="technical_invalid"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),
                            "traceback":traceback.format_exc()}
    finally:
        for handle in handles:handle.remove()
        if torch.cuda.is_available():torch.cuda.synchronize(device)
        receipt["counts"]=dict(counts)
        receipt["internal_seconds"]=time.monotonic()-started
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["artifact_bytes"]=sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"]=os.getpid()
        write_new(OUT/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"counts":counts,
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_raw_complete","technical failure; no retry")


def readback():
    a,old_a,p=checked()
    require(not (OUT/"readback.json").exists(),"cold readback already exists")
    receipt=json.loads((OUT/"receipt.json").read_text())
    outer=json.loads((OUT/"outer.json").read_text())
    require(receipt["status"]=="candidate_raw_complete" and
            receipt["protocol"]==bind(PROTOCOL) and
            receipt["admission"]==bind(ADMISSION) and
            receipt["repair"]==bind(REPAIR) and receipt["fixture"]==bind(FIXTURE) and
            receipt["failed_attempt_outer"]==bind(ORIGINAL_OUT/"outer.json") and
            receipt["preflight"]==bind(PREFLIGHT) and
            receipt["producer"]==bind(Path(__file__)) and
            [x["arm"] for x in receipt["cells"]]==list(ARMS) and
            outer["terminal"] and outer["returncode"]==0 and
            outer["child_pid"]==receipt["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists(),
            "terminal receipt/order/outer differs")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold reader loaded language model")
    batch,raw,trace,sr,planning=prior.prior.source(q,old_a,torch.device("cpu"))
    require(sr["input_identity"]==p["source_input_identity"] and
            int(q.tokenizer.pad_token_id)==p["pad_id"] and
            sorted(q.tokenizer.all_special_ids)==p["special_ids"],
            "cold source/tokenizer differs")
    fixture=base.ConfigOnlyRope();baselines={};vectors={};summaries=[]
    for name,entry in zip(ARMS,receipt["cells"],strict=True):
        require(bind(entry["raw"]["path"])==entry["raw"] and
                bind(entry["inputs"]["path"])==entry["inputs"],
                "cold raw/input binding changed")
        stored={k:torch.tensor(v,dtype=torch.long) for k,v in
                json.loads(Path(entry["inputs"]["path"]).read_text()).items()}
        payload=torch.load(entry["raw"]["path"],map_location="cpu",weights_only=True)
        full=verify_serialized(fixture,batch,raw,p["pad_id"],a,old_a,
                               name,entry,stored,payload)
        logits=payload["logits"]
        require(entry["chosen"]==int(torch.argmax(logits[TARGET]).item()),
                "cold argmax differs")
        err=reference(a,name,full,logits)
        require(entry["accepted_reference_max_error"]==err and
                entry["max_headout_error"]==max(x["oracle_error"] for x in payload["headouts"]),
                "cold reference/headout error differs")
        state_check(name,payload,baselines)
        role="native_A" if name in ("native_anchor","identity_reconstruction") else "coordinate_A"
        require(entry["source_trace_parity"]==prior.prior.trace_check(logits,trace,raw,role,6),
                "cold source/companion trace differs")
        if name in ("native_anchor","current_mask_anchor"):baselines[name]=payload
        vectors[name]=logits[TARGET].double()
        top=torch.topk(vectors[name],2);z=torch.logsumexp(vectors[name],-1)
        summaries.append({"arm":name,"raw":entry["raw"],"inputs":entry["inputs"],
                          "reference_max_error":err,"max_headout_error":entry["max_headout_error"],
                          "winner":int(top.indices[0]),"runner_up":int(top.indices[1]),
                          "top2_logits":top.values.tolist(),
                          "remaining_mass_by_layer":
                            [x["remaining_mass"].tolist() for x in payload["headouts"]],
                          "descriptive_tokens":{
                            str(k):{"logit":float(vectors[name][k]),
                                    "probability":float(torch.exp(vectors[name][k]-z))}
                            for k in a["descriptive_tokens"]}})
        if name not in ("native_anchor","current_mask_anchor"):
            del payload
    require(all(receipt["counts"][k]==a["counts"][k] for k in
                ("model_forwards","vision_forwards","generated_tokens","reused_calls")) and
            len(receipt["cells"])==6,"cold finite count differs")
    probs={k:torch.softmax(v,-1) for k,v in vectors.items()}
    def tv(x,y):return float(torch.abs(probs[x]-probs[y]).sum()/2)
    B=tv("native_anchor","current_mask_anchor")
    distances={k:{"to_native":tv(k,"native_anchor"),
                  "to_current_mask":tv(k,"current_mask_anchor")} for k in ARMS}
    guard=float(a["metrics"]["guard"])
    rH=distances["remove_H"]["to_current_mask"]/B if B>guard else None
    rR=distances["redistribute_R"]["to_current_mask"]/B if B>guard else None
    if B<=guard:
        primary=comparator=False;decision=a["low_baseline_category"]
    else:
        require(rH is not None and rR is not None,"ratio missing")
        near=any(abs(x)<=guard for x in (rH-.5,rR-.5,rR-rH-.1,rH-rR-.1))
        primary=rH<a["primary"]["r_H_max_strict"] and \
                rR-rH>a["primary"]["r_R_minus_r_H_min_strict"]
        comparator=rR<a["comparator"]["r_R_max_strict"] and \
                   rH-rR>a["comparator"]["r_H_minus_r_R_min_strict"]
        decision=(a["numerical_guard_category"] if near else
                  a["primary"]["name"] if primary else
                  a["comparator"]["name"] if comparator else a["other_category"])
    result={"status":"candidate_cold_readback_passed", "protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"repair":bind(REPAIR),"fixture":bind(FIXTURE),
            "failed_attempt_outer":bind(ORIGINAL_OUT/"outer.json"),
            "preflight":bind(PREFLIGHT),
            "producer":bind(Path(__file__)),"receipt":bind(OUT/"receipt.json"),
            "outer":bind(OUT/"outer.json"),"cells":summaries,"counts":receipt["counts"],
            "B":B,"TV_distances":distances,"r_H":rH,"r_R":rR,
            "primary_pass":primary,"comparator_pass":comparator,"decision":decision,
            "outer_seconds":outer["outer_seconds"],
            "cumulative_sequence_gpu_hours":p["attempt_prior_gpu_hours"]+
                                            outer["outer_seconds"]/3600}
    write_new(OUT/"readback.json",result)
    print(json.dumps({"status":result["status"],"B":B,"r_H":rH,"r_R":rR,
                      "decision":decision,"counts":receipt["counts"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","launch","run","readback"))
    {"preflight":preflight,"launch":launch,"run":run,"readback":readback}[
        parser.parse_args().action]()


if __name__=="__main__":main()
