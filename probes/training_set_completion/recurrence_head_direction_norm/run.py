"""One fixed current-query attention decomposition, full prefix, six calls."""
from __future__ import annotations

import argparse
import ast
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

from probes.training_set_completion.recurrence_current_read_decomposition import run as prior


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-head-direction-norm"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
RULING = UNIT / "lead-repair-qualification-v2.json"
PREFLIGHT = UNIT / "supporting/attempt-002-preflight.json"
FIXTURE = UNIT / "supporting/attempt-002-device-qualification.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-head-direction-norm/attempt-002")
FIXTURE_OUT = OUT.parent / "qualification-v2"
FIXTURE_FULL = FIXTURE_OUT / "full-parent-seconds.txt"
MODEL_FULL = OUT / "full-parent-seconds.txt"
SHAS = {PROTOCOL:"60ccbb639cb1a86b0c663edc0454d7806089a703a9c2d51f72fef130d3d979c0",
        ADMISSION:"db2ead15457ee36ba2d7d1d732aa9aa0e451ae6ee0fd9157491f69914cfca317",
        RULING:"24e6a7d8883f2bade152204e6b326fee25150c2384cc204f18e03261a0cd4a25"}
ARMS = ("native_anchor","remove_anchor","native_identity",
        "remove_identity","gain_control","direction_control")
TARGET,C,WIDTH,H0,H1,LAYERS,QH,KVH,DIM,TOL,HEAD_REL = 2,1376,1377,1362,1371,28,16,8,128,2e-4,5e-5
KEYS = prior.KEYS
base = prior.base
history = prior.prior.prior
require,bind,write_new = base.require,base.bind,base.write_new


def contract():
    for path,sha in SHAS.items():require(bind(path)["sha256"]==sha,f"contract changed: {path}")
    a=json.loads(ADMISSION.read_text())
    ruling=json.loads(RULING.read_text())
    require(ruling["status"]=="lead-authorized-fixture-shape-repair-and-conditional-attempt002" and
            ruling["protocol"]==bind(PROTOCOL) and ruling["admission"]==bind(ADMISSION) and
            ruling["fresh_paths"]=={"raw_root":str(OUT),"fixture_root":str(FIXTURE_OUT),
                "preflight":str(PREFLIGHT),"fixture_receipt":str(FIXTURE),
                "candidate":str(UNIT/"candidate-attempt-002-results.md")} and
            ruling["accounting"]["charged_before_continuation_gpu_hours"]==
                0.9370399767159082,"repair authority/paths/accounting changed")
    for key in ("failure_packet","failed_preflight","failed_fixture_plan",
                "failed_fixture_receipt","failed_fixture_stdout","lead_CPU_verification"):
        require(bind(ruling[key]["path"])==ruling[key],f"failed/lead evidence changed: {key}")
    for item in ruling["preserved_sources"]:
        require(bind(item["maintained"]["path"])==item["maintained"] or
                Path(item["maintained"]["path"]) in (Path(__file__),Path(__file__).with_name("device_fixture.py")),
                "unexpected prior maintained source change")
        require(bind(item["capture"]["path"])==item["capture"],
                "retained failed source changed")
    source=Path(__file__).read_text()
    seen=set()
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in ruling["frozen_computational_AST_SHA256"]:
            require(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==
                    ruling["frozen_computational_AST_SHA256"][node.name],
                    f"frozen computational AST changed: {node.name}")
            seen.add(node.name)
    require(seen==set(ruling["frozen_computational_AST_SHA256"]),
            "frozen computational AST missing")
    require(a["status"]=="lead-admitted-six-call-head-direction-norm" and
            a["worker_thread"]=="01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"]=="gpt-6-sol" and a["worker_effort"]=="xhigh" and
            [x["name"] for x in a["cells"]]==list(ARMS) and
            [x["write_formula"] for x in a["cells"]]==[None,"R","N","R","G","D"] and
            [x["mask_mode"] for x in a["cells"]]==["native"]*6 and
            a["qualification"]["headout_relative_bound"]==HEAD_REL and
            a["qualification"]["full_vocabulary_max_abs_error"]==TOL and
            a["counts"]["model_forwards"]==a["counts"]["vision_forwards"]==6 and
            a["counts"]["generated_tokens"]==a["counts"]["reused_calls"]==0 and
            a["counts"]["max_nonmodel_CUDA_fixture_invocations"]==1 and
            a["planning"]["artifact_envelope_bytes"]==3*1024**3 and
            a["owned_paths"]["raw_root"]==str(OUT.parent/"attempt-001") and
            a["owned_paths"]["fixture_root"]==str(FIXTURE_OUT.parent/"qualification-v1") and
            a["qualification"]["per_head_norm_relative_error"]==2e-6 and
            a["qualification"]["per_head_unit_direction_max_abs_error"]==2e-6,
            "admitted cells/gates/cost changed")
    for name in ("protocol","cpu_feasibility","cpu_bindings","cpu_acceptance",
                 "predecessor_acceptance","predecessor_result","predecessor_admission",
                 "predecessor_repair","predecessor_receipt","predecessor_readback","predecessor_outer",
                 "predecessor_preflight","predecessor_verification","predecessor_producer",
                 "source_panel"):
        require(bind(a[name]["path"])==a[name],f"binding changed: {name}")
    for item in a["counterexamples"].values():
        require(bind(item["path"])==item,"counterexample changed")
    for name in ("raw","trace","runtime_receipt","image"):
        require(bind(a["source_bindings"][name]["path"])==a["source_bindings"][name],
                f"source changed: {name}")
    cross=a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"])==cross["maintained"] and
            cross["maintained"]["sha256"]==cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"])==cross["prior_acceptance"],
            "loader crosswalk changed")
    predecessor_a,old_a=prior.contract()
    require(a["source_bindings"]["raw"]==predecessor_a["source_bindings"]["raw"] and
            a["source_bindings"]["trace"]==predecessor_a["source_bindings"]["trace"] and
            a["conditioning"]["current_prefix"]==predecessor_a["conditioning"]["current_prefix"] and
            a["model_geometry"]["query_heads"]==QH and
            a["model_geometry"]["kv_heads"]==KVH and
            a["model_geometry"]["head_dim"]==DIM and
            a["model_geometry"]["target_query_physical"]==C and
            a["model_geometry"]["history_keys_physical"]==[H0,H1],
            "source/current geometry changed")
    for name in ("N","R"):
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
    require(name in ARMS,"unknown fixed cell")
    return prior.mask_for(full,old_a,"native_anchor")


def nonselected_hash(tensor):
    return prior.nonselected_hash(tensor)


def components(q,k,v,cos,sin,native_row,*,target=TARGET,query=C,
               history=(H0,H1),group=2,formula="H+R"):
    return prior.components(q,k,v,cos,sin,native_row,target=target,query=query,
                            history=history,group=group,formula=formula)


def control_metrics(N,R,chosen,kind):
    require(kind in ("G","D") and N.shape==R.shape==chosen.shape==(QH,DIM) and
            N.dtype==R.dtype==chosen.dtype==torch.float64 and
            all(torch.isfinite(x).all().item() for x in (N,R,chosen)),
            "control shape/dtype/finiteness changed")
    n=torch.linalg.vector_norm(N,dim=-1);r=torch.linalg.vector_norm(R,dim=-1)
    require(bool((n>0).all()) and bool((r>0).all()) and
            torch.isfinite(n).all().item() and torch.isfinite(r).all().item(),
            "zero/nonfinite native or removal head norm")
    target=r if kind=="G" else n
    direction=N/n[:,None] if kind=="G" else R/r[:,None]
    cast=chosen.float()
    require(torch.isfinite(cast).all().item(),"FP32 control unrepresentable")
    cast_norm=torch.linalg.vector_norm(cast.double(),dim=-1)
    require(torch.isfinite(cast_norm).all().item() and bool((cast_norm>0).all()),
            "FP32 control zero/nonfinite norm")
    relative=(cast_norm-target).abs()/target
    direction_error=(cast.double()/cast_norm[:,None]-direction).abs().amax(dim=-1)
    require(float(relative.max())<=2e-6 and float(direction_error.max())<=2e-6,
            "per-head control norm/direction gate failed")
    return {"norm_N":n,"norm_R":r,"gain_G":r/n,"gain_D":n/r,
            "relative_error":relative,"direction_error":direction_error}


def controls(vals):
    N=vals["H+R"];R=vals["R"]
    require(N.shape==R.shape==(QH,DIM) and
            N.dtype==R.dtype==torch.float64,"own head-output shapes changed")
    n=torch.linalg.vector_norm(N,dim=-1);r=torch.linalg.vector_norm(R,dim=-1)
    require(torch.isfinite(n).all().item() and torch.isfinite(r).all().item() and
            bool((n>0).all()) and bool((r>0).all()),"zero/nonfinite head norm")
    G=(r/n)[:,None]*N;D=(n/r)[:,None]*R
    g=control_metrics(N,R,G,"G");d=control_metrics(N,R,D,"D")
    return {"N":N,"R":R,"G":G,"D":D}, {
        "norm_N":g["norm_N"],"norm_R":g["norm_R"],
        "gain_G":g["gain_G"],"gain_D":g["gain_D"],
        "G_relative_error":g["relative_error"],
        "D_relative_error":d["relative_error"],
        "G_direction_error":g["direction_error"],
        "D_direction_error":d["direction_error"]}


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
                formulas,norms=controls(vals)
                before=original[TARGET,C].reshape(QH,DIM)
                oracle=formulas["N"]
                error=float((before.double()-oracle).abs().max())
                scale=max(1.0,float(before.abs().max()),float(oracle.abs().max()))
                require(error<=HEAD_REL*scale,
                        f"actual SDPA headout reconstruction failed layer {idx}: {error}>{HEAD_REL*scale}")
                formula=spec(active["admission"],active["name"])["write_formula"]
                chosen=before if formula is None else formulas[formula].to(dtype=original.dtype)
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
                     "control_norms":{key:value.detach().cpu().clone()
                                      for key,value in norms.items()},
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
    actual=native
    require(all(torch.equal(stored[k].detach().cpu(),full[k].detach().cpu()) for k in KEYS) and
            entry["input_hashes"]==history.input_hashes(full,mask) and
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
        formulas,norms=controls(vals)
        before=row["pre_selected"]
        oracle=formulas["N"]
        error=float((before.double()-oracle).abs().max())
        scale=max(1.0,float(before.abs().max()),float(oracle.abs().max()))
        formula=spec(a,name)["write_formula"]
        expected=before if formula is None else formulas[formula].float()
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
                float((mass["history_mass"]-row["history_mass"]).abs().max())<=1e-12 and
                set(row["control_norms"])==set(norms) and
                all(row["control_norms"][key].shape==(QH,) and
                    row["control_norms"][key].dtype==torch.float64 and
                    torch.isfinite(row["control_norms"][key]).all().item() and
                    float((row["control_norms"][key]-value).abs().max())<=
                    (1e-12 if key in ("norm_N","norm_R","gain_G","gain_D") else
                     2*torch.finfo(torch.float32).eps)*
                    max(1.0,float(value.abs().max()),
                        float(row["control_norms"][key].abs().max()))
                    for key,value in norms.items()) and
                all(float(row["control_norms"][key].max())<=2e-6 for key in
                    ("G_relative_error","D_relative_error",
                     "G_direction_error","D_direction_error")),
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
                f"fixed history/header/companion state changed: {key}")
    require(float((payload["current_by_layer"][:,:5]-
                   native["current_by_layer"][:,:5]).abs().max())<=TOL and
            max(float((payload["logits"][i]-native["logits"][i]).abs().max())
                for i in (0,1,3))<=TOL,"companion full vectors changed")
    key={"native_identity":"native_anchor","remove_identity":"remove_anchor"}.get(name)
    if key:
        require(float((payload["logits"]-baselines[key]["logits"]).abs().max())<=TOL,
                "fresh identity full-vector failed")


def synthetic_entry(a,old_a,full,name,seed):
    mask,native,selected=mask_for(full,old_a,name)
    actual=native
    entry={"arm":name,"order":ARMS.index(name),"step":6,
           "input_hashes":history.input_hashes(full,mask),
           "actual_mask_hash":base.tensor_hash(actual),
           "actual_layers":list(range(LAYERS)),
           "actual_layer_mask_hashes":[base.tensor_hash(actual)]*LAYERS,
           "selected_cells":int(selected.sum()),
           "selected_native_hash":base.tensor_hash(native[selected]),
           "selected_actual_hash":base.tensor_hash(actual[selected]),
           "complement_native_hash":base.tensor_hash(native[~selected]),
           "complement_actual_hash":base.tensor_hash(actual[~selected]),
           "headout_complement_hashes":[x["post_complement_hash"] for x in seed["headouts"]]}
    return entry


def synthetic_payload(a,name,seed,native):
    rows=[];native_row=native[TARGET,0,C].detach().cpu()
    for i,(data,saved) in enumerate(zip(seed["qkv"],seed["headouts"],strict=True)):
        vals,mass=components(data["Q"],data["K"],data["V"],
                             seed["rotary_cos"],seed["rotary_sin"],native_row)
        formulas,norms=controls(vals)
        before=saved["pre_selected"].clone()
        oracle=formulas["N"]
        selected=spec(a,name)["write_formula"]
        post=before.clone() if selected is None else formulas[selected].float()
        rows.append({"layer":i,"pre_selected":before,"post_selected":post,
                     "oracle_error":float((before.double()-oracle).abs().max()),
                     "oracle_scale":max(1.,float(before.abs().max()),float(oracle.abs().max())),
                     "remaining_mass":mass["remaining_mass"].clone(),
                     "history_mass":mass["history_mass"].clone(),
                     "control_norms":{k:v.clone() for k,v in norms.items()},
                     "pre_complement_hash":saved["pre_complement_hash"],
                     "post_complement_hash":saved["post_complement_hash"],
                     "observed_post":True})
    return {**seed,"arm":name,"actual_mask":native.detach().cpu().clone(),"headouts":rows}


def cpu_checks(a,old_a,batch,raw,pad):
    from copy import deepcopy
    from transformers.masking_utils import create_causal_mask
    fixture=base.ConfigOnlyRope()
    full=full_input(fixture,batch,raw,pad,a)
    cfg=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    cfg._attn_implementation="sdpa"
    native=base.native_4d(full["attention_mask"])
    actual=create_causal_mask(cfg,torch.empty((4,WIDTH,1)),full["attention_mask"],
                              full["cache_position"],None,position_ids=full["position_ids"][0])
    require(torch.equal(actual,native),"installed native SDPA mask changed")
    checks=["installed_native_SDPA_mask"]
    for name in ARMS:
        mask,n,selected=mask_for(full,old_a,name)
        require(torch.equal(mask,full["attention_mask"]) and torch.equal(n,native) and
                int(selected.sum())==0,"native mask/rectangle changed")
        checks.append("native_mask_"+name)
    # The maintained mask caller must reject non-native rectangles.
    for label,kwargs in (("wrong_query",{"spans":[[1375,1376]]}),
                         ("wrong_key",{"key":(1363,1371)}),
                         ("wrong_target",{"target":3})):
        try:prior.prior.mask_for(full,old_a,"native_anchor",**kwargs)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError(label)
    q=torch.randn((QH,DIM),generator=torch.Generator().manual_seed(34))*.1
    k=torch.randn((WIDTH,KVH,DIM),generator=torch.Generator().manual_seed(35))*.1
    v=torch.randn((WIDTH,KVH,DIM),generator=torch.Generator().manual_seed(36))*.1
    cos=torch.ones((WIDTH,DIM));sin=torch.zeros_like(cos)
    vals,mass=components(q,k,v,cos,sin,native[TARGET,0,C])
    formulas,norms=controls(vals)
    require(bool((mass["remaining_mass"]>0).all()),"partition mass changed")
    for kind,target in (("G",norms["norm_R"]),("D",norms["norm_N"])):
        require(float((torch.linalg.vector_norm(formulas[kind],dim=-1)-target).abs().max())<1e-12,
                "per-head norm formula changed")
    checks.append("per_head_N_R_G_D_FP64_and_FP32")
    rq=q.unsqueeze(0).unsqueeze(2)
    rk=k.permute(1,0,2).repeat_interleave(2,0).unsqueeze(0)
    rv=v.permute(1,0,2).repeat_interleave(2,0).unsqueeze(0)
    sd=F.scaled_dot_product_attention(rq,rk,rv,
          attn_mask=native[TARGET,0,C].view(1,1,1,WIDTH),dropout_p=0.0,scale=DIM**-.5)[0,:,0]
    require(float((sd-vals["H+R"]).abs().max())<HEAD_REL,"CPU SDPA oracle differs")
    checks.append("actual_CPU_SDPA_headout")
    for label,kw in (("query",{"query":C-1}),("target",{"target":1}),
                     ("history",{"history":(H0+1,H1)}),("group",{"group":1}),
                     ("formula",{"formula":"H-R"})):
        try:components(q,k,v,cos,sin,native[TARGET,0,C],**kw)
        except ValueError:checks.append("reject_component_"+label)
        else:raise AssertionError(label)
    for label,x,y,chosen,kind in (("zero_N",torch.zeros_like(formulas["N"]),formulas["R"],formulas["G"],"G"),
                                  ("zero_R",formulas["N"],torch.zeros_like(formulas["R"]),formulas["D"],"D"),
                                  ("global_gain",formulas["N"],formulas["R"],formulas["N"]*2,"G"),
                                  ("swapped_G_D",formulas["N"],formulas["R"],formulas["D"],"G")):
        try:control_metrics(x,y,chosen,kind)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError(label)
    # Exercise the actual installed writer/observer on one CPU attention layer.
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
    att=fake.model.language_model.layers[0].self_attn
    probe=torch.zeros((4,WIDTH,QH*DIM));probe[TARGET,C]=formulas["N"].float().reshape(-1)
    def exercise(active):
        fake.model.language_model.rotary_emb((cos.repeat(4,1,1),sin.repeat(4,1,1)))
        xq=torch.zeros((4,WIDTH,QH,DIM));xq[TARGET,C]=q
        xk=torch.zeros((4,WIDTH,KVH,DIM));xk[TARGET]=k
        xv=torch.zeros((4,WIDTH,KVH*DIM));xv[TARGET]=v.reshape(WIDTH,-1)
        att.q_norm(xq);att.k_norm(xk);att.v_proj(xv)
        return att(probe,attention_mask=native,past_key_values=None,
                   cache_position=full["cache_position"])
    for name in ARMS:
        active={"admission":a,"name":name,"qkv":[{}],"rotary":None,
                "masks":[],"layers":[],"headouts":[],"native_mask":native,
                "expected_mask":native,"full":full}
        with hooks(fake,active):got=exercise(active)
        formula=spec(a,name)["write_formula"]
        expected=probe[TARGET,C] if formula is None else formulas[formula].float().reshape(-1)
        require(torch.equal(got[TARGET,C],expected) and
                active["headouts"][0]["observed_post"] and
                not att.o_proj._forward_pre_hooks and not att._forward_pre_hooks,
                "actual CPU writer/consumer failed: "+name)
        checks.append("actual_CPU_writer_consumer_"+name)
    try:
        active.update(qkv=[{}],rotary=None,masks=[],layers=[],headouts=[])
        with hooks(fake,active):
            exercise(active)
            raise RuntimeError("forced body exception")
    except RuntimeError as exc:require(str(exc)=="forced body exception","wrong forced exception")
    require(not att.o_proj._forward_pre_hooks and not att._forward_pre_hooks,
            "exception left hook installed")
    checks.append("forced_exception_restoration")
    seed=torch.load(a["references"]["N"]["raw"]["path"],map_location="cpu",weights_only=True)
    stored={key:full[key].detach().cpu().clone() for key in KEYS}
    for name in ARMS:
        entry=synthetic_entry(a,old_a,full,name,seed)
        payload=synthetic_payload(a,name,seed,native)
        verify_serialized(fixture,batch,raw,pad,a,old_a,name,entry,stored,payload)
        checks.append("actual_serialized_CPU_"+name)
        if name=="gain_control":reference_entry,reference_payload=entry,payload
    def reject(label,entry=reference_entry,stored=stored,payload=reference_payload):
        try:verify_serialized(fixture,batch,raw,pad,a,old_a,
                              "gain_control",entry,stored,payload)
        except ValueError:checks.append("reject_reader_"+label)
        else:raise AssertionError("actual reader accepted "+label)
    bad=deepcopy(reference_entry);bad["order"]=0;reject("wrong_order",entry=bad)
    bad=deepcopy(reference_entry);bad["arm"]="direction_control";reject("wrong_arm",entry=bad)
    for label,key,index in (("history","input_ids",(TARGET,H0)),
                            ("own","input_ids",(TARGET,C)),
                            ("companion","input_ids",(0,C)),
                            ("position","position_ids",(0,TARGET,C)),
                            ("source_mask","attention_mask",(TARGET,C))):
        bad={**stored,key:stored[key].clone()};bad[key][index]+=1
        reject(label,stored=bad)
    for label,index in (("selected_history",(TARGET,0,C,H0)),
                        ("current_to_current",(TARGET,0,C,C)),
                        ("wrong_target",(3,0,C,H0)),
                        ("wrong_query",(TARGET,0,C-1,H0))):
        bad={**reference_payload,"actual_mask":reference_payload["actual_mask"].clone()}
        bad["actual_mask"][index]=~bad["actual_mask"][index]
        reject(label,payload=bad)
    for label,key,delta in (("formula","post_selected",1e-4),
                            ("oracle","oracle_error",1e-7),
                            ("mass","remaining_mass",1e-7)):
        rows=[dict(x) for x in reference_payload["headouts"]]
        rows[0][key]=rows[0][key]+delta
        reject(label,payload={**reference_payload,"headouts":rows})
    rows=[dict(x) for x in reference_payload["headouts"]]
    rows[0]["control_norms"]=dict(rows[0]["control_norms"])
    rows[0]["control_norms"]["gain_G"]=rows[0]["control_norms"]["gain_G"]+1e-6
    reject("gain_record",payload={**reference_payload,"headouts":rows})
    rows=[dict(x) for x in reference_payload["headouts"]]
    rows[0]["post_complement_hash"]="0"*64
    reject("complement",payload={**reference_payload,"headouts":rows})
    return checks


def preflight():
    a,old_a=contract()
    require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists() and
            not FIXTURE.exists(),"fixed attempt/fixture already prepared")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preparation loaded language model")
    batch,raw,trace,sr,planning=history.source(q,old_a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id)
    require(list(batch.request_ids)==a["source"]["request_ids"] and
            sr["input_identity"]==json.loads(Path(a["source_bindings"]["runtime_receipt"]["path"]).read_text())["input_identity"] and
            int(batch.inputs["pixel_values"].numel())==24502272 and
            [len(x["token_ids"]) for x in raw]==[255,37,3084,3084] and
            list(map(len,batch.prompt_token_ids))==[1336,1362,1362,1320],
            "original source/geometry differs")
    checks=cpu_checks(a,old_a,batch,raw,pad)
    from probes.training_set_completion.recurrence_head_direction_norm import device_fixture
    saved=torch.load(a["references"]["N"]["raw"]["path"],map_location="cpu",weights_only=True)
    stored={k:torch.tensor(v,dtype=torch.long) for k,v in
            json.loads(Path(a["references"]["N"]["inputs"]["path"]).read_text()).items()}
    cold_checks=device_fixture.cold_paths(a,old_a,batch,raw,pad,saved,stored)
    require(cold_checks==["cold_2D_source_and_4D_native_mask"]+
            ["cold_CPU_"+name for name in ARMS],
            "actual repaired cold fixture path failed")
    full=full_input(base.ConfigOnlyRope(),batch,raw,pad,a)
    for key in ("N","R"):
        saved=json.loads(Path(a["references"][key]["inputs"]["path"]).read_text())
        require(all(saved[k]==full[k].tolist() for k in KEYS),
                "distinct original reference input differs")
    require(full["input_ids"].shape==(4,WIDTH) and len(checks)>=30 and
            a["planning"]["planning_25pct_payload_allowance_bytes"]<
            a["planning"]["artifact_envelope_bytes"],
            "CPU caller or capacity forecast failed")
    previous=json.loads(prior.PREFLIGHT.read_text())
    sources=[Path(__file__),Path(__file__).with_name("device_fixture.py"),
             Path(prior.__file__),Path(prior.prior.__file__),Path(history.__file__),
             Path(history.old.__file__),Path(base.__file__),
             Path(inspect.getfile(qmod)),Path(inspect.getfile(sdpa)),
             Path(inspect.getfile(AutoConfig))]
    sources += [Path(x["maintained"]["path"]) for x in previous["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(sources):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("external")/path.name
        saved=base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    module="probes.training_set_completion.recurrence_head_direction_norm"
    command=["python","-B","-m",module+".run"]
    fixture_command=(f"/usr/bin/time -f '%e' -o {FIXTURE_FULL} "
                     f"python -B -m {module}.device_fixture "
                     f"> {FIXTURE_OUT/'parent-stdout.log'} 2>&1")
    model_command=(f"/usr/bin/time -f '%e' -o {MODEL_FULL} "
                   f"python -B -m {module}.run launch "
                   f"> {OUT/'parent-stdout.log'} 2>&1")
    packet={"status":"cpu_qualified_fixture_pending","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"producer":bind(Path(__file__)),
            "ruling":bind(RULING),
            "failed_fixture_receipt":bind(UNIT/"supporting/attempt-001-device-qualification.json"),
            "charged_prior_gpu_hours":json.loads(RULING.read_text())["accounting"]["charged_before_continuation_gpu_hours"],
            "fixture_source":bind(Path(__file__).with_name("device_fixture.py")),
            "source_identity":sr["identity"],"source_input_identity":sr["input_identity"],
            "request_ids":list(batch.request_ids),
            "prompt_lengths":list(map(len,batch.prompt_token_ids)),
            "raw_lengths":[len(x["token_ids"]) for x in raw],
            "pad_id":pad,"special_ids":sorted(q.tokenizer.all_special_ids),
            "pixel_elements":int(batch.inputs["pixel_values"].numel()),"width":WIDTH,
            "image_grids":[list(x) for x in batch.image_grids],
            "cpu_checks":checks,"actual_cold_fixture_checks":cold_checks,
            "direct_source_captures":captures,
            "forecast_outer_seconds":a["planning"]["two_x_planned_model_outer_seconds"],
            "artifact_forecast_bytes":a["planning"]["planning_25pct_payload_allowance_bytes"],
            "commands":{"preflight":command+["preflight"],
                        "fixture_enclosing":fixture_command,
                        "model_enclosing":model_command,
                        "launch":command+["launch"],"model":command+["run"],
                        "cold":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"cpu_checks":len(checks),
                      "source_captures":len(captures),"width":WIDTH,
                      "pixel_elements":packet["pixel_elements"],
                      "artifact_forecast_bytes":packet["artifact_forecast_bytes"]}))


def checked(*,need_fixture=True):
    a,old_a=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_fixture_pending" and
            p["protocol"]==bind(PROTOCOL) and p["admission"]==bind(ADMISSION) and
            p["ruling"]==bind(RULING) and
            p["failed_fixture_receipt"]==
                bind(UNIT/"supporting/attempt-001-device-qualification.json") and
            p["charged_prior_gpu_hours"]==
                json.loads(RULING.read_text())["accounting"]["charged_before_continuation_gpu_hours"] and
            p["producer"]==bind(Path(__file__)) and
            p["fixture_source"]==bind(Path(__file__).with_name("device_fixture.py")) and
            p["width"]==WIDTH and p["pixel_elements"]==24502272 and
            len(p["cpu_checks"])>=30 and
            p["actual_cold_fixture_checks"]==
                ["cold_2D_source_and_4D_native_mask"]+
                ["cold_CPU_"+name for name in ARMS] and
            p["artifact_forecast_bytes"]<a["planning"]["artifact_envelope_bytes"],
            "frozen CPU preflight/producer differs")
    for item in p["direct_source_captures"]:
        require(bind(item["maintained"]["path"])==item["maintained"] and
                bind(item["capture"]["path"])==item["capture"],
                "direct maintained source/capture changed")
    if need_fixture:
        f=json.loads(FIXTURE.read_text())
        require(f["status"]=="qualified_nonmodel_CUDA_GREEN" and
                f["protocol"]==bind(PROTOCOL) and f["admission"]==bind(ADMISSION) and
                f["ruling"]==bind(RULING) and
                f["preflight"]==bind(PREFLIGHT) and
                f["producer"]==bind(Path(__file__)) and
                f["fixture_source"]==bind(Path(__file__).with_name("device_fixture.py")) and
                f["model_loads"]==f["model_forwards"]==f["vision_forwards"]==
                    f["generated_tokens"]==0 and
                f["outer_seconds"]>0 and f["returncode"]==0 and f["terminal"] and
                not Path(f"/proc/{f['child_pid']}").exists(),
                "bounded mixed-device fixture not qualified")
        require(FIXTURE_FULL.exists() and
                math.isfinite(float(FIXTURE_FULL.read_text().strip())) and
                float(FIXTURE_FULL.read_text().strip())>=f["outer_seconds"],
                "full-parent fixture cost missing/shorter than internal interval")
    return a,old_a,p


def launch():
    checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"outer.json").exists(),
            "fixed attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-B","-m",
             "probes.training_set_completion.recurrence_head_direction_norm.run","run"]
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
             "ruling":bind(RULING),"failed_fixture_receipt":p["failed_fixture_receipt"],
             "fixture":bind(FIXTURE),"preflight":bind(PREFLIGHT),
             "producer":bind(Path(__file__)),"counts":counts,"cells":[]}
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
        batch,raw,trace,sr,planning=history.source(q,old_a,device)
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
            active["actual_input"]=history.input_hashes(kwargs,kwargs["attention_mask"])
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
                if name=="gain_control":
                    require(len(receipt["cells"])==4 and
                            [x["arm"] for x in receipt["cells"]]==list(ARMS[:4]) and
                            counts["model_forwards"]==counts["vision_forwards"]==4,
                            "four actual references/identity gates not qualified")
                full=full_input(model,batch,raw,p["pad_id"],a)
                mask,native,selected=mask_for(full,old_a,name)
                actual=native
                active.clear();active.update(admission=a,name=name,full=full,
                    expected_mask=actual,native_mask=native,selected=selected,
                    expected_input_hashes=history.input_hashes(full,mask),
                    actual_input=None,qkv=[{} for _ in range(LAYERS)],
                    rotary=None,masks=[],layers=[],headouts=[],history=[],current=[],companions=[])
                with hooks(model,active):out=model(**{**full,"attention_mask":mask})
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
                entry=history.entry_for(full,mask,native,selected,active,
                           int(torch.argmax(logits[TARGET]).item()),6,rawpath,inputpath)
                entry.update(arm=name,order=ARMS.index(name),
                    headout_complement_hashes=[x["post_complement_hash"] for x in active["headouts"]],
                    max_headout_error=max(x["oracle_error"] for x in active["headouts"]),
                    max_G_relative_error=max(float(x["control_norms"]["G_relative_error"].max()) for x in active["headouts"]),
                    max_D_relative_error=max(float(x["control_norms"]["D_relative_error"].max()) for x in active["headouts"]),
                    max_G_direction_error=max(float(x["control_norms"]["G_direction_error"].max()) for x in active["headouts"]),
                    max_D_direction_error=max(float(x["control_norms"]["D_direction_error"].max()) for x in active["headouts"]))
                receipt["cells"].append(entry)
                stored={k:torch.tensor(v,dtype=torch.long) for k,v in
                        json.loads(inputpath.read_text()).items()}
                verify_serialized(model,batch,raw,p["pad_id"],a,old_a,name,entry,stored,payload)
                entry["accepted_reference_max_error"]=reference(a,name,full,logits)
                state_check(name,payload,baselines)
                role="native_A" if name in ("native_anchor","native_identity") else "coordinate_A"
                entry["source_trace_parity"]=history.trace_check(logits,trace,raw,role,6)
                if name in ("native_anchor","remove_anchor"):baselines[name]=payload
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
            receipt["ruling"]==bind(RULING) and
            receipt["failed_fixture_receipt"]==p["failed_fixture_receipt"] and
            receipt["fixture"]==bind(FIXTURE) and
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
    batch,raw,trace,sr,planning=history.source(q,old_a,torch.device("cpu"))
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
        role="native_A" if name in ("native_anchor","native_identity") else "coordinate_A"
        require(entry["source_trace_parity"]==history.trace_check(logits,trace,raw,role,6),
                "cold source/companion trace differs")
        if name in ("native_anchor","remove_anchor"):baselines[name]=payload
        vectors[name]=logits[TARGET].double()
        top=torch.topk(vectors[name],2);z=torch.logsumexp(vectors[name],-1)
        summaries.append({"arm":name,"raw":entry["raw"],"inputs":entry["inputs"],
                          "reference_max_error":err,"max_headout_error":entry["max_headout_error"],
                          "max_G_relative_error":entry["max_G_relative_error"],
                          "max_D_relative_error":entry["max_D_relative_error"],
                          "max_G_direction_error":entry["max_G_direction_error"],
                          "max_D_direction_error":entry["max_D_direction_error"],
                          "winner":int(top.indices[0]),"runner_up":int(top.indices[1]),
                          "top2_logits":top.values.tolist(),
                          "remaining_mass_by_layer":
                            [x["remaining_mass"].tolist() for x in payload["headouts"]],
                          "descriptive_tokens":{
                            str(k):{"logit":float(vectors[name][k]),
                                    "probability":float(torch.exp(vectors[name][k]-z))}
                            for k in a["descriptive_tokens"]}})
        del payload
    require(all(receipt["counts"][k]==a["counts"][k] for k in
                ("model_forwards","vision_forwards","generated_tokens","reused_calls")) and
            len(receipt["cells"])==6,"cold finite count differs")
    probs={k:torch.softmax(v,-1) for k,v in vectors.items()}
    def tv(x,y):return float(torch.abs(probs[x]-probs[y]).sum()/2)
    B=tv("native_anchor","remove_anchor")
    distances={k:{"to_N":tv(k,"native_anchor"),"to_R":tv(k,"remove_anchor")}
               for k in ARMS}
    all_pairwise={x:{y:tv(x,y) for y in ARMS} for x in ARMS}
    guard=float(a["metrics"]["guard"])
    ratios={"d_N_G":distances["gain_control"]["to_N"]/B if B>guard else None,
            "d_R_G":distances["gain_control"]["to_R"]/B if B>guard else None,
            "d_N_D":distances["direction_control"]["to_N"]/B if B>guard else None,
            "d_R_D":distances["direction_control"]["to_R"]/B if B>guard else None}
    if B<=guard:
        primary=comparator=False;decision=a["low_baseline_category"]
    else:
        gN,gR,dN,dR=(ratios[k] for k in ("d_N_G","d_R_G","d_N_D","d_R_D"))
        primary=(gN<.5 and gR-gN>.1 and dR<.5 and dN-dR>.1)
        comparator=(gR<.5 and gN-gR>.1 and dN<.5 and dR-dN>.1)
        near=any(abs(x)<=guard for x in (gN-.5,gR-.5,dN-.5,dR-.5,
                gR-gN-.1,gN-gR-.1,dN-dR-.1,dR-dN-.1))
        decision=(a["numerical_guard_category"] if near else
                  a["primary"]["name"] if primary else
                  a["comparator"]["name"] if comparator else a["other_category"])
    fixture_receipt=json.loads(FIXTURE.read_text())
    fixture_full=float(FIXTURE_FULL.read_text().strip())
    model_full=float(MODEL_FULL.read_text().strip())
    require(math.isfinite(model_full) and model_full>=outer["outer_seconds"],
            "full-parent model cost missing/shorter than internal interval")
    result={"status":"candidate_cold_readback_passed","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"ruling":bind(RULING),
            "fixture":bind(FIXTURE),
            "fixture_full_parent":bind(FIXTURE_FULL),
            "model_full_parent":bind(MODEL_FULL),
            "preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
            "receipt":bind(OUT/"receipt.json"),"outer":bind(OUT/"outer.json"),
            "cells":summaries,"counts":receipt["counts"],"B":B,
            "TV_distances":distances,"TV_pairwise":all_pairwise,"ratios":ratios,
            "TV_G_D":tv("gain_control","direction_control"),
            "primary_pass":primary,"comparator_pass":comparator,"decision":decision,
            "fixture_internal_seconds":fixture_receipt["outer_seconds"],
            "fixture_full_parent_seconds":fixture_full,
            "model_internal_seconds":outer["outer_seconds"],
            "model_full_parent_seconds":model_full,
            "cumulative_sequence_gpu_hours":p["charged_prior_gpu_hours"]+
                (fixture_full+model_full)/3600}
    write_new(OUT/"readback.json",result)
    print(json.dumps({"status":result["status"],"B":B,"ratios":ratios,
                      "decision":decision,"counts":receipt["counts"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","launch","run","readback"))
    {"preflight":preflight,"launch":launch,"run":run,"readback":readback}[
        parser.parse_args().action]()


if __name__=="__main__":main()
