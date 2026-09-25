"""Single non-model CPU/CUDA boundary check for the fixed six serialized paths."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from probes.training_set_completion.recurrence_head_direction_norm import run as m

PLAN=m.FIXTURE_OUT/"plan.json"
CHILD=m.FIXTURE_OUT/"child.json"
LOG=m.FIXTURE_OUT/"stdout.log"


def cold_paths(a,old_a,batch,raw,pad,saved,stored):
    cold=m.base.ConfigOnlyRope()
    cpu_full=m.full_input(cold,batch,raw,pad,a)
    source_mask,cpu_mask,_=m.mask_for(cpu_full,old_a,"native_anchor")
    m.require(source_mask.shape==(4,m.WIDTH) and
              cpu_mask.shape==(4,1,m.WIDTH,m.WIDTH),
              "cold source/native mask dimensionality changed")
    checks=["cold_2D_source_and_4D_native_mask"]
    for name in m.ARMS:
        entry=m.synthetic_entry(a,old_a,cpu_full,name,saved)
        trial=m.synthetic_payload(a,name,saved,cpu_mask)
        m.verify_serialized(cold,batch,raw,pad,a,old_a,name,entry,stored,trial)
        checks.append("cold_CPU_"+name)
    return checks


def child():
    a,old_a,p=m.checked(need_fixture=False)
    plan=json.loads(PLAN.read_text())
    m.require(plan["preflight"]==m.bind(m.PREFLIGHT) and
              plan["ruling"]==m.bind(m.RULING) and
              plan["producer"]==m.bind(Path(m.__file__)) and
              plan["fixture_source"]==m.bind(Path(__file__)),
              "fixture plan/source changed")
    q=m.base.load_qwen_components_from_options(m.base.QwenLoadOptions(
        base_model=str(m.base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    m.require(q.model is None,"fixture loaded language model")
    batch,raw,trace,sr,planning=m.history.source(q,old_a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id)
    m.require(sr["input_identity"]==p["source_input_identity"],"fixture source changed")
    saved=torch.load(a["references"]["N"]["raw"]["path"],map_location="cpu",weights_only=True)
    stored={k:torch.tensor(v,dtype=torch.long) for k,v in
            json.loads(Path(a["references"]["N"]["inputs"]["path"]).read_text()).items()}
    checks=cold_paths(a,old_a,batch,raw,pad,saved,stored)
    class DeviceConfig(m.base.ConfigOnlyRope):
        def __init__(self):
            super().__init__();self.anchor=torch.empty(0,device="cuda:0")
        def parameters(self):return iter((self.anchor,))
    m.require(torch.cuda.is_available(),"CUDA fixture unavailable")
    anchored=DeviceConfig()
    full=m.full_input(anchored,batch,raw,pad,a)
    mask,native,_=m.mask_for(full,old_a,"native_anchor")
    placements={k:str(full[k].device) for k in m.KEYS}
    placements.update(actual_mask=str(native.device),saved_mask=str(saved["actual_mask"].device),
                      saved_Q=str(saved["qkv"][0]["Q"].device),
                      serialized_input=str(stored["input_ids"].device))
    m.require(all(placements[k]=="cuda:0" for k in m.KEYS) and
              placements["actual_mask"]=="cuda:0" and
              placements["saved_mask"]==placements["saved_Q"]==
              placements["serialized_input"]=="cpu",
              "actual CUDA/CPU fixture placement differs")
    for name in m.ARMS:
        entry=m.synthetic_entry(a,old_a,full,name,saved)
        trial=m.synthetic_payload(a,name,saved,native)
        m.verify_serialized(anchored,batch,raw,pad,a,old_a,name,entry,stored,trial)
        checks.append("mixed_device_actual_reader_"+name)
        if name=="gain_control":reference_entry,reference_payload=entry,trial
    def reject(label,entry=reference_entry,stored=stored,payload=reference_payload):
        try:m.verify_serialized(anchored,batch,raw,pad,a,old_a,
                                "gain_control",entry,stored,payload)
        except ValueError:checks.append("reject_"+label)
        else:raise AssertionError("actual mixed-device reader accepted "+label)
    for label,key,index in (("history_input","input_ids",(m.TARGET,m.H0)),
                            ("own_input","input_ids",(m.TARGET,m.C)),
                            ("companion_input","input_ids",(0,m.C)),
                            ("position","position_ids",(0,m.TARGET,m.C)),
                            ("source_mask","attention_mask",(m.TARGET,m.C))):
        bad={**stored,key:stored[key].clone()};bad[key][index]+=1
        reject(label,stored=bad)
    for label,index in (("selected_mask",(m.TARGET,0,m.C,m.H0)),
                        ("current_to_current",(m.TARGET,0,m.C,m.C)),
                        ("wrong_target",(3,0,m.C,m.H0)),
                        ("wrong_query",(m.TARGET,0,m.C-1,m.H0))):
        bad={**reference_payload,"actual_mask":reference_payload["actual_mask"].clone()}
        bad["actual_mask"][index]=~bad["actual_mask"][index]
        reject(label,payload=bad)
    bad={**reference_entry,"order":0};reject("order",entry=bad)
    bad={**reference_entry,"arm":"direction_control"};reject("arm",entry=bad)
    for label,axis in (("Q","Q"),("K","K"),("V","V")):
        qkv=list(reference_payload["qkv"]);qkv[0]=dict(qkv[0]);qkv[0][axis]=qkv[0][axis].clone()
        qkv[0][axis]+=1000
        reject(label,payload={**reference_payload,"qkv":qkv})
    bad={**reference_payload,"rotary_cos":reference_payload["rotary_cos"].clone()}
    bad["rotary_cos"][m.C,0]+=.1;reject("rotary",payload=bad)
    for label,key,delta in (("formula","post_selected",1e-4),
                            ("oracle","oracle_error",1e-7),
                            ("mass","remaining_mass",1e-7)):
        rows=[dict(x) for x in reference_payload["headouts"]]
        rows[0][key]=rows[0][key]+delta
        reject(label,payload={**reference_payload,"headouts":rows})
    rows=[dict(x) for x in reference_payload["headouts"]]
    rows[0]["control_norms"]=dict(rows[0]["control_norms"])
    rows[0]["control_norms"]["gain_G"]=rows[0]["control_norms"]["gain_G"]+1e-6
    reject("norm_record",payload={**reference_payload,"headouts":rows})
    rows=[dict(x) for x in reference_payload["headouts"]]
    rows[0]["post_complement_hash"]="0"*64
    reject("complement",payload={**reference_payload,"headouts":rows})
    result={"status":"child_pass","plan":m.bind(PLAN),
            "protocol":m.bind(m.PROTOCOL),"admission":m.bind(m.ADMISSION),
            "ruling":m.bind(m.RULING),
            "preflight":m.bind(m.PREFLIGHT),"producer":m.bind(Path(m.__file__)),
            "fixture_source":m.bind(Path(__file__)),"placements":placements,
            "checks":checks,"synthetic_only":True,
            "model_loads":0,"model_forwards":0,"vision_forwards":0,
            "generated_tokens":0,"pid":os.getpid()}
    m.write_new(CHILD,result)
    print(json.dumps({"status":result["status"],"checks":len(checks),
                      "placements":placements}))


def parent():
    a,old_a,p=m.checked(need_fixture=False)
    m.require(not PLAN.exists() and not CHILD.exists() and not LOG.exists() and
              not m.FIXTURE.exists(),"fixture already invoked")
    m.FIXTURE_OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-B","-m",
             "probes.training_set_completion.recurrence_head_direction_norm.device_fixture",
             "--child"]
    plan={"status":"frozen_before_nonmodel_CUDA","protocol":m.bind(m.PROTOCOL),
          "admission":m.bind(m.ADMISSION),"preflight":m.bind(m.PREFLIGHT),
          "ruling":m.bind(m.RULING),
          "producer":m.bind(Path(m.__file__)),"fixture_source":m.bind(Path(__file__)),
          "command":command}
    m.write_new(PLAN,plan)
    start=time.monotonic()
    with LOG.open("x") as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
        code=process.wait()
    duration=time.monotonic()-start
    packet={"status":"qualified_nonmodel_CUDA_GREEN" if code==0 and CHILD.exists() else
                     "technical_invalid","protocol":m.bind(m.PROTOCOL),
            "admission":m.bind(m.ADMISSION),"preflight":m.bind(m.PREFLIGHT),
            "ruling":m.bind(m.RULING),
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
