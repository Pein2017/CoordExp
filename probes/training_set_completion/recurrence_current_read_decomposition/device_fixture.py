"""One bounded non-model CPU/CUDA serialized-verifier qualification."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from probes.training_set_completion.recurrence_current_read_decomposition import run as m


PLAN=m.UNIT/"supporting/attempt-002-device-plan.json"
CHILD=m.UNIT/"supporting/attempt-002-device-child.json"
LOG=m.UNIT/"supporting/attempt-002-device-stdout.log"


def child():
    plan=json.loads(PLAN.read_text())
    m.require(plan["ruling"]==m.bind(m.REPAIR) and
              plan["producer"]==m.bind(Path(m.__file__)) and
              plan["fixture_source"]==m.bind(Path(__file__)) and
              plan["failed_raw"]==m.bind(m.ORIGINAL_OUT/"native_anchor.pt") and
              plan["failed_inputs"]==m.bind(m.ORIGINAL_OUT/"inputs-native_anchor.json"),
              "fixture plan/source changed")
    a,old_a=m.contract()
    q=m.base.load_qwen_components_from_options(m.base.QwenLoadOptions(
        base_model=str(m.base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    m.require(q.model is None,"fixture loaded language model")
    batch,raw,trace,sr,planning=m.prior.prior.source(q,old_a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id)
    checks=m.cpu_checks(a,old_a,batch,raw,pad)
    source=m.ORIGINAL_OUT
    original=json.loads((source/"receipt.json").read_text())["cells"][0]
    stored={k:torch.tensor(v,dtype=torch.long) for k,v in
            json.loads((source/"inputs-native_anchor.json").read_text()).items()}
    payload=torch.load(source/"native_anchor.pt",map_location="cpu",weights_only=True)
    m.verify_serialized(m.base.ConfigOnlyRope(),batch,raw,pad,a,old_a,
                        "native_anchor",original,stored,payload)
    checks.append("original_saved_payload_cold_CPU_28_layers_unqualified")

    class DeviceConfig(m.base.ConfigOnlyRope):
        def __init__(self):
            super().__init__()
            self.anchor=torch.empty(0,device="cuda:0")
        def parameters(self):
            return iter((self.anchor,))

    m.require(torch.cuda.is_available(),"CUDA fixture device unavailable")
    anchored=DeviceConfig()
    full=m.full_input(anchored,batch,raw,pad,a)
    mask,native,_=m.mask_for(full,old_a,"native_anchor")
    placements={k:str(full[k].device) for k in ("input_ids","attention_mask",
                                               "position_ids","cache_position")}
    placements.update(reconstructed_mask=str(native.device),
                      saved_mask=str(payload["actual_mask"].device),
                      saved_qkv=str(payload["qkv"][0]["Q"].device))
    m.require(all(x=="cuda:0" for x in (*placements.values(),)[:5]) and
              placements["saved_mask"]==placements["saved_qkv"]=="cpu" and
              all(x.device.type=="cpu" for x in stored.values()),
              "real CUDA/CPU fixture placement differs")
    m.verify_serialized(anchored,batch,raw,pad,a,old_a,
                        "native_anchor",original,stored,payload)
    checks.append("original_saved_payload_CUDA_reconstruction_CPU_saved_28_layers_unqualified")

    # All six are synthetic caller/serialized-path checks, never model endpoints.
    native_cpu=native[m.TARGET,0,m.C].detach().cpu()
    math_rows=[]
    for data in payload["qkv"]:
        math_rows.append(m.components(data["Q"],data["K"],data["V"],
                                      payload["rotary_cos"],payload["rotary_sin"],native_cpu))
    synthetic={}
    for name in m.ARMS:
        chosen_mask,nmask,selected=m.mask_for(full,old_a,name)
        actual=nmask if name!="current_mask_anchor" else chosen_mask
        formula=m.spec(a,name)["write_formula"]
        rows=[]
        for i,((vals,mass),saved) in enumerate(zip(math_rows,payload["headouts"],strict=True)):
            before=(vals["O_R"].float() if name=="current_mask_anchor" else
                    saved["pre_selected"])
            oracle=vals["O_R" if name=="current_mask_anchor" else "H+R"]
            rows.append({"layer":i,"pre_selected":before.clone(),
                         "post_selected":(before if formula is None else vals[formula].float()).clone(),
                         "oracle_error":float((before.double()-oracle).abs().max()),
                         "oracle_scale":max(1.0,float(before.abs().max()),float(oracle.abs().max())),
                         "remaining_mass":mass["remaining_mass"].clone(),
                         "history_mass":mass["history_mass"].clone(),
                         "pre_complement_hash":saved["pre_complement_hash"],
                         "post_complement_hash":saved["post_complement_hash"],
                         "observed_post":True})
        trial={**payload,"arm":name,"actual_mask":actual.detach().cpu().clone(),
               "headouts":rows}
        entry={**original,"arm":name,"order":m.ARMS.index(name),
               "input_hashes":m.prior.prior.input_hashes(full,chosen_mask),
               "actual_mask_hash":m.base.tensor_hash(actual),
               "actual_layer_mask_hashes":[m.base.tensor_hash(actual)]*m.LAYERS,
               "selected_cells":int(selected.sum()),
               "selected_native_hash":m.base.tensor_hash(nmask[selected]),
               "selected_actual_hash":m.base.tensor_hash(actual[selected]),
               "complement_native_hash":m.base.tensor_hash(nmask[~selected]),
               "complement_actual_hash":m.base.tensor_hash(actual[~selected]),
               "headout_complement_hashes":[x["post_complement_hash"] for x in rows]}
        m.verify_serialized(anchored,batch,raw,pad,a,old_a,name,entry,stored,trial)
        synthetic[name]=(entry,trial)
        checks.append("synthetic_actual_GPU_reader_"+name)

    from copy import copy
    entry,trial=synthetic["remove_H"]
    def reject(label,entry2=entry,stored2=stored,payload2=trial):
        try:m.verify_serialized(anchored,batch,raw,pad,a,old_a,
                                "remove_H",entry2,stored2,payload2)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError("actual GPU verifier accepted "+label)
    for label,key,index in (("history_input","input_ids",(m.TARGET,m.H0)),
                            ("own_input","input_ids",(m.TARGET,m.C)),
                            ("companion_input","input_ids",(0,m.C)),
                            ("position","position_ids",(0,m.TARGET,m.C)),
                            ("source_mask","attention_mask",(m.TARGET,m.C))):
        bad={**stored,key:stored[key].clone()};bad[key][index]+=1
        reject(label,stored2=bad)
    for label,index in (("selected_mask",(m.TARGET,0,m.C,m.H0)),
                        ("current_to_current",(m.TARGET,0,m.C,m.C)),
                        ("wrong_target",(3,0,m.C,m.H0)),
                        ("wrong_query",(m.TARGET,0,m.C-1,m.H0)),
                        ("wrong_history_key",(m.TARGET,0,m.C,m.H1))):
        bad={**trial,"actual_mask":trial["actual_mask"].clone()}
        bad["actual_mask"][index]=~bad["actual_mask"][index]
        reject(label,payload2=bad)
    bad={**entry,"order":0};reject("cell_order",entry2=bad)
    bad={**entry,"arm":"redistribute_R"};reject("reference_alias",entry2=bad)
    bad={**trial,"headouts":[dict(x) for x in trial["headouts"]]}
    bad["headouts"][0],bad["headouts"][1]=bad["headouts"][1],bad["headouts"][0]
    reject("capture_order",payload2=bad)
    for axis in ("Q","K","V"):
        qkv=list(trial["qkv"]);qkv[0]=dict(qkv[0]);qkv[0][axis]=qkv[0][axis].clone()
        if axis=="K":
            qkv[0][axis][m.C]=trial["qkv"][0]["Q"].reshape(m.KVH,2,m.DIM)[:,0]*1000
        else:
            qkv[0][axis]+=1000
        reject("wrong_"+axis,payload2={**trial,"qkv":qkv})
    bad={**trial,"rotary_cos":trial["rotary_cos"].clone()}
    bad["rotary_cos"][m.C,0]+=.1
    reject("wrong_rotary",payload2=bad)
    for label,key,delta in (("mass","remaining_mass",1e-7),
                            ("oracle_error","oracle_error",1e-7),
                            ("oracle_scale","oracle_scale",1e-7),
                            ("wrong_formula_FP32","post_selected",1e-4)):
        rows=[dict(x) for x in trial["headouts"]]
        rows[0][key]=(rows[0][key]+delta)
        reject(label,payload2={**trial,"headouts":rows})
    rows=[dict(x) for x in trial["headouts"]]
    rows[0]["post_complement_hash"]="0"*64
    reject("unselected_complement",payload2={**trial,"headouts":rows})
    for label,kw in (("target",{"target":3}),("query",{"query":m.C-1}),
                     ("history",{"history":(m.H0+1,m.H1)})):
        try:m.components(payload["qkv"][0]["Q"],payload["qkv"][0]["K"],
                         payload["qkv"][0]["V"],payload["rotary_cos"],
                         payload["rotary_sin"],native_cpu,**kw)
        except ValueError:checks.append("reject_"+label+"_geometry")
        else:raise AssertionError("component accepted "+label)
    result={"status":"child_pass","ruling":m.bind(m.REPAIR),
            "producer":m.bind(Path(m.__file__)),"fixture_source":m.bind(Path(__file__)),
            "plan":m.bind(PLAN),"placements":placements,"checks":checks,
            "synthetic_only":True,"saved_attempt001_unqualified":True,
            "model_loads":0,"model_forwards":0,"vision_forwards":0,"generated_tokens":0,
            "pid":os.getpid()}
    m.write_new(CHILD,result)
    print(json.dumps({"status":result["status"],"checks":len(checks),"placements":placements}))


def parent():
    a,old_a=m.contract()
    m.require(not PLAN.exists() and not CHILD.exists() and not LOG.exists() and
              not m.FIXTURE.exists(),"fixture invocation already exists")
    m.OUT.mkdir(parents=True,exist_ok=True)
    source_captures=[]
    for path in (Path(m.__file__),Path(__file__)):
        saved=m.base.preserve_source(path,run_root=m.OUT,
                    relative_name=path.relative_to(m.REPO))
        source_captures.append({"maintained":m.bind(path),"capture":m.bind(saved)})
    plan={"status":"frozen_before_nonmodel_CUDA","ruling":m.bind(m.REPAIR),
          "protocol":m.bind(m.PROTOCOL),"admission":m.bind(m.ADMISSION),
          "producer":m.bind(Path(m.__file__)),"fixture_source":m.bind(Path(__file__)),
          "failed_raw":m.bind(m.ORIGINAL_OUT/"native_anchor.pt"),
          "failed_inputs":m.bind(m.ORIGINAL_OUT/"inputs-native_anchor.json"),
          "source_captures":source_captures,
          "command":[sys.executable,"-B","-m",
                     "probes.training_set_completion.recurrence_current_read_decomposition.device_fixture",
                     "--child"]}
    m.write_new(PLAN,plan)
    start=time.monotonic()
    with LOG.open("x") as output:
        process=subprocess.Popen(plan["command"],stdout=output,stderr=subprocess.STDOUT)
        code=process.wait()
    duration=time.monotonic()-start
    packet={"status":"qualified_nonmodel_CUDA_GREEN" if code==0 and CHILD.exists() else
                     "technical_invalid","ruling":m.bind(m.REPAIR),
            "producer":m.bind(Path(m.__file__)),"fixture_source":m.bind(Path(__file__)),
            "plan":m.bind(PLAN),"stdout":m.bind(LOG),
            "child":m.bind(CHILD) if CHILD.exists() else None,
            "outer_seconds":duration,"returncode":code,"child_pid":process.pid,
            "terminal":True,"model_loads":0,"model_forwards":0,
            "vision_forwards":0,"generated_tokens":0}
    m.write_new(m.FIXTURE,packet)
    print(json.dumps({"status":packet["status"],"outer_seconds":duration,
                      "returncode":code,"child_pid":process.pid}))
    m.require(packet["status"]=="qualified_nonmodel_CUDA_GREEN",
              "bounded CUDA fixture failed; no second invocation")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--child",action="store_true")
    (child if parser.parse_args().child else parent)()
