"""Ten full-prefix one-record read-stage arms; no cached model state."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_collapse_history_content import run as old
from probes.training_set_completion.recurrence_free_header_routing.run import parse_row
from src.data.geometry import iou_xyxy


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-history-read-stage"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
REPAIR = UNIT / "lead-repair-attempt-002.json"
FIXTURE_V2 = UNIT / "lead-fixture-correction-v2.json"
QUALIFICATION = UNIT / "supporting/lead-device-qualification-v3.json"
FAILED_PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
FAILED_OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-history-read-stage/attempt-001")
PREFLIGHT = UNIT / "supporting/attempt-002-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-history-read-stage/attempt-002")
SHAS = {PROTOCOL: "76c1fabfbc05be19f97492888704805f6516c1810789d625812e14df3c23c5c3",
        ADMISSION: "c4e9e3d856ea8a95e4e9d561ceb012a284caa7a47b1cf88b3d3e899ddcfbb56b",
        REPAIR: "ac8d04b9a85e5dd8c7e74b6153126555d40b3c48889651feef9178702363f1bc",
        FIXTURE_V2: "83712753b1779fdf3323fd437b6e0c664608b644cb268ea11a84c7867d79f28c",
        QUALIFICATION: "a0ae4e2a29e25326c33df3fb058b1525e291100465d77de9ce91ef816fce6c73"}
ARMS = ("native_A", "native_F", "sham_A", "sham_F", "all_A", "all_F",
        "header_A", "header_F", "coordinate_A", "coordinate_F")
TARGET, OFFSET, PROMPT, SPLIT, TOL = 2, 9, 1362, 1375, 2e-4
KEYS = ("input_ids", "attention_mask", "position_ids", "cache_position")
base = old.base
require, bind, write_new = base.require, base.bind, base.write_new


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    repair = json.loads(REPAIR.read_text())
    qualified = json.loads(QUALIFICATION.read_text())
    source = Path(__file__).read_text()
    verifier = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "verify_serialized")
    require(hashlib.sha256(ast.get_source_segment(source, verifier).encode()).hexdigest() ==
            qualified["verify_serialized_source_sha256"] ==
            "86e6b4d9d39e432bde84eb2820b6e196cb13ad1f974f52b7924ea610bbcfe536",
            "qualified serialized verifier changed")
    require(qualified["status"] == "lead-accepted-device-boundary-qualification-only" and
            repair["attempt002"]["raw_root"] == qualified["required_new_paths"]["raw_root"] == str(OUT) and
            repair["attempt002"]["preflight"] == qualified["required_new_paths"]["preflight"] == str(PREFLIGHT) and
            qualified["required_new_paths"]["candidate_report"] ==
                str(UNIT / "candidate-attempt-002-complete-results.md") and
            qualified["accounting"]["cumulative_before_model_attempt002_gpu_hours"] ==
                0.7885676063376733 and
            repair["attempt002"]["arms"] == a["arms"] and
            repair["attempt002"]["counts"]["max_model_forwards"] == 118,
            "repair authority/attempt accounting changed")
    for name in ("takeover", "parent_repair", "fixture_addendum", "red_plan", "green_plan",
                 "red", "green", "red_outer", "green_outer", "fixture"):
        require(bind(qualified[name]["path"]) == qualified[name],
                f"device qualification binding changed: {name}")
    for name in ("failure_packet", "failed_preflight", "failed_candidate",
                 "immutable_failed_producer_capture"):
        require(bind(repair[name]["path"]) == repair[name], f"failed attempt binding changed: {name}")
    require(all(bind(x["path"]) == x for x in repair["failed_raw_tree"]) and
            bind(FAILED_OUT / "outer.json")["sha256"] ==
                next(x["sha256"] for x in repair["failed_raw_tree"] if x["path"] == str(FAILED_OUT / "outer.json")),
            "failed raw/charge changed")
    require(a["status"] == "lead-admitted-finite-history-read-stage" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            [x["name"] for x in a["arms"]] == list(ARMS) and
            [x["max_tokens"] for x in a["arms"]] == [9]*6+[16]*4 and
            a["source"]["target_index"] == TARGET and
            a["source"]["history_raw_end"] == OFFSET and
            a["source"]["A_tokens"] == list(old.ROW0) and
            a["source"]["F_tokens"] == list(old.ROW1) and
            a["mask_geometry"]["history_keys_physical"] == [PROMPT, PROMPT+OFFSET] and
            a["mask_geometry"]["first_coordinate_query_physical"] == SPLIT and
            a["counts"]["max_model_forwards"] == a["counts"]["max_vision_forwards"] ==
            a["counts"]["max_emitted_tokens"] == 118 and
            a["planning"]["artifact_envelope_bytes"] == 4*1024**3 and
            a["owned_paths"][-1] == str(FAILED_OUT), "admitted scope changed")
    for name in ("protocol", "cpu_acceptance", "x1_acceptance", "one_record_acceptance",
                 "one_record_readback", "one_record_receipt", "feasibility_report",
                 "reference_bindings", "source_panel"):
        require(bind(a[name]["path"]) == a[name], f"admission binding changed: {name}")
    for name in ("raw", "trace", "runtime_receipt", "image"):
        require(bind(a["source_bindings"][name]["path"]) == a["source_bindings"][name],
                f"source binding changed: {name}")
    cross = a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"]) == cross["maintained"] and
            cross["maintained"]["sha256"] == cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"]) == cross["prior_acceptance"],
            "loader crosswalk changed")
    old_a = old.contract()
    require(a["source_bindings"]["raw"] == old_a["source_bindings"]["raw"] and
            a["source_bindings"]["trace"] == old_a["source_bindings"]["trace"] and
            a["source_bindings"]["source_identity"] == old_a["source_bindings"]["source_identity"],
            "one-record source changed")
    refs = json.loads(Path(a["reference_bindings"]["path"]).read_text())
    require(set(refs["accepted_references"]) == {"native_A","native_F","masked_A","masked_F"},
            "reference set changed")
    for row in refs["accepted_references"].values():
        require(len(row["steps"]) == 9 and row["stop"] == "complete", "reference row missing")
        for step in row["steps"]:
            for kind in ("raw", "inputs"):
                require(bind(step[kind]["path"]) == step[kind], "reference vector/input changed")
    return a, refs, old_a


def spec(a, arm):
    require(arm in ARMS, "unknown arm")
    return a["arms"][ARMS.index(arm)]


def validate_arm_order(arms):
    require([x["arm"] for x in arms] == list(ARMS), "missing/reordered arm receipt")


def source(q, a, device):
    batch, raw, trace, receipt, planning = old.source(q, old.contract(), device)
    require(list(batch.request_ids) == a["source"]["request_ids"] and
            receipt["identity"]["attention"] == "sdpa" and
            int(batch.inputs["pixel_values"].numel()) == 24502272,
            "source batch/model differs")
    return batch, raw, trace, receipt, planning


def step_inputs(model, batch, raw, pad, a, arm, emitted, *, previous=None,
                target=TARGET, prefix=OFFSET):
    s = spec(a, arm); t = len(emitted)
    require(target == TARGET and prefix == OFFSET and 0 <= t < s["max_tokens"] and
            (t == 0 or emitted[-1] == previous), "target/prefix/own-token/step changed")
    tails = base._prefix_tokens(raw, OFFSET+t, pad)
    history = list(old.ROW0 if s["history"] == "A" else old.ROW1)
    tails[TARGET] = history + list(emitted)
    histories = [list(p)+tail for p,tail in zip(batch.prompt_token_ids,tails,strict=True)]
    full = base.exact_history_inputs(model,batch.inputs,histories,pad_token_id=pad,logits_to_keep=1)
    width = int(full["input_ids"].shape[1]);full["cache_position"] = torch.arange(width,device=full["input_ids"].device)
    require(width == PROMPT+OFFSET+t and full["attention_mask"].shape == (4,width) and
            full["position_ids"].shape == (3,4,width) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            [i for i,(x,y) in enumerate(zip(tails[TARGET][:OFFSET],old.ROW0)) if x!=y] ==
            ([] if s["history"]=="A" else [5,6,7]) and
            all(tails[i] == raw[i]["token_ids"][:OFFSET+t] for i in (0,1,3)) and
            full["attention_mask"][:,PROMPT:].all().item() and
            all(int(full["attention_mask"][i,:PROMPT].sum()) == len(batch.prompt_token_ids[i]) for i in range(4)),
            "history/companions/positions/source changed")
    return full


def mask_for(full, a, arm, *, target=TARGET, key=None, query=None, split=SPLIT,
             mode=None, source_hash=None):
    s = spec(a, arm); width = int(full["input_ids"].shape[1]);t=width-PROMPT-OFFSET
    qspan=(PROMPT+OFFSET,width);kspan=(PROMPT,PROMPT+OFFSET)
    require(target == TARGET and 0 <= t <= 15 and split == SPLIT and
            key in (None,kspan) and query in (None,qspan) and
            mode in (None,s["mask_mode"]) and
            (source_hash is None or base.tensor_hash(full["attention_mask"]) == source_hash),
            "wrong target/key/query/split/mode/source attention")
    native=base.native_4d(full["attention_mask"])
    selected=torch.zeros_like(native)
    if s["mask_mode"] in ("all","header","coordinate"):
        lo,hi={"all":qspan,"header":(qspan[0],min(split,qspan[1])),
               "coordinate":(split,qspan[1])}[s["mask_mode"]]
        selected[TARGET,0,lo:hi,kspan[0]:kspan[1]]=True
    count={"native":0,"sham":0,"all":9*t,"header":9*min(t,4),
           "coordinate":9*max(t-4,0)}[s["mask_mode"]]
    require(int(selected.sum()) == count and (count == 0 or bool(native[selected].all())),
            "selected unreadable/wrong rectangle")
    actual=native.clone();actual[selected]=False
    verify_rectangle(native,actual,selected,count)
    return full["attention_mask"] if s["mask_mode"]=="native" else actual,native,selected


def verify_rectangle(native,actual,selected,count):
    require(native.shape == actual.shape == selected.shape and int(selected.sum()) == count and
            torch.equal(actual[~selected],native[~selected]) and
            torch.equal(actual[selected],torch.zeros_like(native[selected])),
            "same-rectangle selected/complement changed")


def caller(model,full,a,arm,seen):
    mask,native,selected=mask_for(full,a,arm,source_hash=base.tensor_hash(full["attention_mask"]))
    actual=native if spec(a,arm)["mask_mode"]=="native" else mask
    seen.update(expected_mask=actual,native_mask=native,selected=selected,
                actual_input=None,layers=[],layer_mask_hashes=[],history=[],companions=[])
    return model(**{**full,"attention_mask":mask})


def input_hashes(full,mask):
    h={k:base.tensor_hash(full[k]) for k in KEYS}
    h["attention_mask"]=base.tensor_hash(mask)
    return h


def verify_online(model,batch,raw,pad,a,arm,emitted,full,seen,logits,chosen):
    expected=step_inputs(model,batch,raw,pad,a,arm,emitted,previous=emitted[-1] if emitted else None)
    mask,native,selected=mask_for(expected,a,arm)
    actual=native if spec(a,arm)["mask_mode"]=="native" else mask
    require(all(torch.equal(full[k],expected[k]) for k in KEYS) and
            seen["actual_input"]==input_hashes(expected,mask) and
            seen["layers"]==list(range(28)) and
            seen["layer_mask_hashes"]==[base.tensor_hash(actual)]*28 and
            torch.equal(seen["expected_mask"],actual) and
            torch.equal(seen["selected"],selected) and
            chosen==int(torch.argmax(logits[TARGET]).item()),
            "actual caller/source/28-layer mask/greedy changed")
    return mask,native,selected


def entry_for(full,mask,native,selected,seen,chosen,t,rawpath,inputpath):
    actual=seen["expected_mask"]
    return {"step":t,"raw":bind(rawpath),"inputs":bind(inputpath),
            "input_hashes":input_hashes(full,mask),"actual_layers":seen["layers"],
            "actual_layer_mask_hashes":seen["layer_mask_hashes"],
            "selected_cells":int(selected.sum()),
            "selected_native_hash":base.tensor_hash(native[selected]),
            "selected_actual_hash":base.tensor_hash(actual[selected]),
            "complement_native_hash":base.tensor_hash(native[~selected]),
            "complement_actual_hash":base.tensor_hash(actual[~selected]),
            "actual_mask_hash":base.tensor_hash(actual),"chosen":chosen}


def verify_serialized(model,batch,raw,pad,a,arm,emitted,entry,stored,payload):
    t=len(emitted);require(entry["step"]==t and payload["arm"]==arm and payload["step"]==t,
                            "serialized arm/step changed")
    expected=step_inputs(model,batch,raw,pad,a,arm,emitted,previous=emitted[-1] if t else None)
    mask,native,selected=mask_for(expected,a,arm)
    actual=native if spec(a,arm)["mask_mode"]=="native" else mask
    require(all(torch.equal(stored[k].detach().cpu(),expected[k].detach().cpu()) for k in KEYS) and
            entry["input_hashes"]==input_hashes(expected,mask) and
            entry["actual_layers"]==list(range(28)) and
            entry["actual_layer_mask_hashes"]==[base.tensor_hash(actual)]*28 and
            torch.equal(payload["actual_mask"].detach().cpu(),actual.detach().cpu()) and
            entry["actual_mask_hash"]==base.tensor_hash(actual) and
            entry["selected_cells"]==int(selected.sum()) and
            entry["selected_native_hash"]==base.tensor_hash(native[selected]) and
            entry["selected_actual_hash"]==base.tensor_hash(actual[selected]) and
            entry["complement_native_hash"]==entry["complement_actual_hash"]==
                base.tensor_hash(native[~selected]) and
            payload["logits"].shape==(4,152670) and torch.isfinite(payload["logits"]).all().item() and
            entry["chosen"]==int(torch.argmax(payload["logits"][TARGET]).item()),
            "serialized mask/input/own prefix/vector differs")
    return expected


def reference(a,refs,arm,t,full,logits,chosen):
    s=spec(a,arm)
    if t>=s["reference_steps"]:return None
    row=refs["accepted_references"][s["reference_key"]];item=row["steps"][t]
    stored=json.loads(Path(item["inputs"]["path"]).read_text())
    require(all(stored[k]==full[k].tolist() for k in KEYS),"accepted same-base input differs")
    old_logits=torch.load(item["raw"]["path"],map_location="cpu",weights_only=True)["logits"]
    err=float((logits-old_logits).abs().max())
    require(err<=TOL and chosen==item["chosen"],"accepted full-vector/chosen reference differs")
    return err


def compare_states(a,arm,t,payload,logits,baselines):
    native=baselines.get("native_"+spec(a,arm)["history"])
    if native is not None:
        old=native[min(t,8)]
        require(float((payload["historical_by_layer"]-old["historical_by_layer"]).abs().max())<=TOL,
                "same-history prior states changed")
        if arm.startswith("sham_"):
            require(t<9 and torch.equal(logits,old["logits"]) and
                    torch.equal(payload["historical_by_layer"],old["historical_by_layer"]) and
                    torch.equal(payload["companions_by_layer"],old["companions_by_layer"]),
                    "independent identity sham differs")
    if arm!="native_A" and t<9:
        first=baselines["native_A"][t]
        require(max(float((payload["companions_by_layer"]-first["companions_by_layer"]).abs().max()),
                    max(float((logits[i]-first["logits"][i]).abs().max()) for i in (0,1,3)))<=TOL,
                "companion states/vectors changed")


def trace_check(logits,trace,raw,arm,t):
    ids=range(4) if arm=="native_A" else (0,1,3)
    records=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
             absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],
             role="read_stage_"+arm) for i in ids]
    require(all(x["passed"] for x in records),"active original source trace changed")
    return records


def cpu_checks(a,batch,raw,pad,special):
    fixture=base.ConfigOnlyRope();cfg=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    cfg._attn_implementation="sdpa";checks=[]
    require(parse_row(list(old.ROW0),special)["stop"]=="complete" and
            parse_row(list(old.ROW1),special)["stop"]=="complete" and
            parse_row([151645],special)["stop"]=="eos" and
            parse_row([151649],special)["stop"]=="early_row_terminator" and
            parse_row([151646,8987,151647,151648,152670],special)["stop"]=="malformed_coordinate" and
            parse_row([151646]+[8987]*15,special)["stop"]=="cap", "parser boundary differs")
    checks.append("parser_complete_eos_early_exclusive_cap")
    for t in (0,1,4,5,15):
        emitted=list(old.ROW1[:t]) if t<=9 else [151646]+[8987]*(t-1)
        for arm in ARMS:
            if t>=spec(a,arm)["max_tokens"]:continue
            full=step_inputs(fixture,batch,raw,pad,a,arm,emitted,previous=emitted[-1] if t else None)
            mode=spec(a,arm)["mask_mode"]
            mask,native,selected=mask_for(full,a,arm)
            installed=create_causal_mask(cfg,torch.empty((*full["attention_mask"].shape,1)),
                full["attention_mask"],full["cache_position"],None,position_ids=full["position_ids"][0])
            require(torch.equal(native,installed),"installed native SDPA mask changed")
            class Fake:
                def __call__(self,**kwargs):
                    seen["actual_input"]=input_hashes(kwargs,kwargs["attention_mask"])
                    z=torch.zeros((4,1,152670));z[TARGET,0,151646]=1
                    return SimpleNamespace(logits=z)
            seen={};logits=caller(Fake(),full,a,arm,seen).logits[:,-1]
            seen["layers"]=list(range(28));seen["layer_mask_hashes"]=[base.tensor_hash(seen["expected_mask"])]*28
            verify_online(fixture,batch,raw,pad,a,arm,emitted,full,seen,logits,151646)
            fakepayload={"arm":arm,"step":t,"actual_mask":seen["expected_mask"],"logits":logits}
            fakeentry=entry_for(full,mask,native,selected,seen,151646,t,Path(__file__),Path(__file__))
            stored={k:full[k].clone() for k in KEYS}
            verify_serialized(fixture,batch,raw,pad,a,arm,emitted,fakeentry,stored,fakepayload)
            checks.append(f"caller_reader_{arm}_t{t}")
            for label,kw in (("target",{"target":3}),("key",{"key":(PROMPT+1,PROMPT+OFFSET)}),
                             ("query",{"query":(PROMPT+OFFSET-1,PROMPT+OFFSET+t)}),
                             ("split",{"split":SPLIT+1}),("mode",{"mode":"bad"})):
                try:mask_for(full,a,arm,**kw)
                except ValueError:checks.append(f"reject_{label}_{arm}_t{t}")
                else:raise AssertionError(label)
            if mode in ("header","coordinate") and t>=5:
                other="coordinate" if mode=="header" else "header"
                try:mask_for(full,a,arm,mode=other)
                except ValueError:checks.append(f"reject_shifted_stage_{arm}_t{t}")
                else:raise AssertionError("shifted stage")
            wrongmask=fakepayload["actual_mask"].clone();wrongmask[0,0,-1,0]=~wrongmask[0,0,-1,0]
            for label,mutpayload,mutstored in (("wrong_complement",{**fakepayload,"actual_mask":wrongmask},stored),
                ("wrong_serialized_prefix",fakepayload,{**stored,"input_ids":stored["input_ids"].clone()})):
                if label=="wrong_serialized_prefix":mutstored["input_ids"][TARGET,-1]+=1
                try:verify_serialized(fixture,batch,raw,pad,a,arm,emitted,fakeentry,mutstored,mutpayload)
                except ValueError:checks.append(f"reject_{label}_{arm}_t{t}")
                else:raise AssertionError(label)
            for label,key,index in (("wrong_companion","input_ids",(0,PROMPT)),
                                    ("wrong_history","input_ids",(TARGET,PROMPT)),
                                    ("wrong_position","position_ids",(0,TARGET,PROMPT))):
                changed={**stored,key:stored[key].clone()};changed[key][index]+=1
                try:verify_serialized(fixture,batch,raw,pad,a,arm,emitted,fakeentry,changed,fakepayload)
                except ValueError:checks.append(f"reject_{label}_{arm}_t{t}")
                else:raise AssertionError(label)
            if t:
                current=fakepayload["actual_mask"].clone()
                current[TARGET,0,PROMPT+OFFSET+t-1,PROMPT+OFFSET+t-1]=False
                try:verify_serialized(fixture,batch,raw,pad,a,arm,emitted,fakeentry,stored,
                                      {**fakepayload,"actual_mask":current})
                except ValueError:checks.append(f"reject_current_to_current_{arm}_t{t}")
                else:raise AssertionError("current-to-current")
            if mode in ("header","coordinate") and t>=5:
                other="coordinate_" if mode=="header" else "header_"
                wrong=mask_for(full,a,other+spec(a,arm)["history"])[0]
                try:verify_serialized(fixture,batch,raw,pad,a,arm,emitted,fakeentry,stored,
                                      {**fakepayload,"actual_mask":wrong})
                except ValueError:checks.append(f"reject_wrong_rectangle_{arm}_t{t}")
                else:raise AssertionError("wrong rectangle")
            if t:
                wrongprefix=list(emitted);wrongprefix[-1]+=1
                try:step_inputs(fixture,batch,raw,pad,a,arm,wrongprefix,previous=emitted[-1])
                except ValueError:checks.append(f"reject_wrong_own_token_{arm}_t{t}")
                else:raise AssertionError("own token")
            if t<=4:
                allmask=mask_for(full,a,"all_"+spec(a,arm)["history"])[1]
                h=mask_for(full,a,"header_"+spec(a,arm)["history"])[0]
                z=mask_for(full,a,"coordinate_"+spec(a,arm)["history"])[0]
                whole=mask_for(full,a,"all_"+spec(a,arm)["history"])[0]
                require(torch.equal(h,whole) and torch.equal(z,allmask),"through-t4 structural identity failed")
    fake=[{"arm":x} for x in ARMS]
    validate_arm_order(fake)
    for changed in (fake[:-1],fake[1:2]+fake[:1]+fake[2:]):
        try:validate_arm_order(changed)
        except ValueError:checks.append("reject_missing_or_reordered_arms")
        else:raise AssertionError("arm order")
    return checks


def preflight():
    a,refs,_=contract();require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),
                                "attempt already prepared")
    qualified=json.loads(QUALIFICATION.read_text())
    failed=json.loads((FAILED_OUT/"receipt.json").read_text())
    failed_outer=json.loads((FAILED_OUT/"outer.json").read_text())
    require(failed["status"]=="technical_invalid" and
            failed["counts"]=={"model_forwards":1,"vision_forwards":1,
                               "emitted_target_tokens":0,"reused_calls":0} and
            failed_outer["terminal"] and failed_outer["returncode"]==1 and
            failed_outer["outer_seconds"]==15.32397124916315 and
            qualified["accounting"]["cumulative_before_model_attempt002_gpu_hours"]==
                0.7885676063376733,
            "failed charge/qualified prior changed")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preparation loaded model")
    batch,raw,trace,sr,planning=source(q,a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    checks=cpu_checks(a,batch,raw,pad,special)
    require(len(checks)>100,"actual caller/reader mutation coverage missing")
    fixture=base.ConfigOnlyRope();widths=[]
    for t in range(16):
        emitted=list(old.ROW1[:t]) if t<=9 else [151646]+[8987]*(t-1)
        full=step_inputs(fixture,batch,raw,pad,a,"coordinate_F",emitted,
                         previous=emitted[-1] if t else None)
        widths.append(int(full["input_ids"].shape[1]))
    pixels=int(batch.inputs["pixel_values"].numel())
    require(widths==list(range(1371,1387)) and pixels==24502272 and
            [len(x["token_ids"]) for x in raw]==[255,37,3084,3084],"source shape changed")
    one=json.loads(Path(a["one_record_receipt"]["path"]).read_text())
    outer=json.loads(Path(a["one_record_receipt"]["path"]).with_name("outer.json").read_text())
    require(one["counts"]["model_forwards"]==45 and outer["returncode"]==0,"measured prior changed")
    oldsq=5*sum(x*x for x in range(1371,1380))
    new_sq=6*sum(x*x for x in range(1371,1380))+4*sum(x*x for x in widths)
    forecast=2*outer["outer_seconds"]*new_sq/oldsq
    artifacts=2*one["artifact_bytes"]*118/45+64*1024**2
    require(artifacts<a["planning"]["artifact_envelope_bytes"] and
            abs(forecast-a["planning"]["maximum_outer_seconds_range"][0])<1e-6,
            "source-shape capacity forecast changed")
    previous=json.loads(old.PREFLIGHT.read_text())
    direct=[Path(__file__),Path(old.__file__),Path(base.__file__),
            Path(inspect.getfile(parse_row)),Path(inspect.getfile(create_causal_mask)),
            Path(inspect.getfile(modeling_qwen3_vl)),Path(inspect.getfile(iou_xyxy))]
    direct += [Path(x["maintained"]["path"]) for x in previous["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(direct):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_history_read_stage.run"]
    packet={"status":"cpu_qualified_before_gpu","admission":bind(ADMISSION),
            "protocol":bind(PROTOCOL),"repair":bind(REPAIR),
            "fixture_addendum":bind(FIXTURE_V2),"device_qualification":bind(QUALIFICATION),
            "failed_preflight":bind(FAILED_PREFLIGHT),
            "failed_receipt":bind(FAILED_OUT/"receipt.json"),
            "failed_outer":bind(FAILED_OUT/"outer.json"),
            "prior_charged_sequence_gpu_hours":0.7885676063376733,
            "producer":bind(Path(__file__)),
            "source_identity":sr["identity"],"source_bindings":a["source_bindings"],
            "reference_bindings":a["reference_bindings"],"request_ids":list(batch.request_ids),
            "prompt_lengths":list(map(len,batch.prompt_token_ids)),
            "raw_lengths":[len(x["token_ids"]) for x in raw],"pad_id":pad,
            "special_ids":sorted(special),"pixel_elements":pixels,
            "image_grids":[list(x) for x in batch.image_grids],"widths":widths,
            "cpu_checks":checks,"forecast_outer_seconds":forecast,
            "artifact_forecast_bytes":artifacts,"direct_source_captures":captures,
            "commands":{"preflight":command+["preflight"],
                        "launch":command+["launch"],"gpu":command+["run"],
                        "readback":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),"captures":len(captures),
                      "widths":[widths[0],widths[-1]],"pixels":pixels,
                      "forecast_outer_seconds":forecast,"artifact_bytes":artifacts}))


def checked():
    a,refs,_=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and
            p["admission"]==bind(ADMISSION) and p["protocol"]==bind(PROTOCOL) and
            p["repair"]==bind(REPAIR) and
            p["fixture_addendum"]==bind(FIXTURE_V2) and
            p["device_qualification"]==bind(QUALIFICATION) and
            p["failed_preflight"]==bind(FAILED_PREFLIGHT) and
            p["failed_receipt"]==bind(FAILED_OUT/"receipt.json") and
            p["failed_outer"]==bind(FAILED_OUT/"outer.json") and
            p["prior_charged_sequence_gpu_hours"]==0.7885676063376733 and
            p["producer"]==bind(Path(__file__)) and
            p["reference_bindings"]==a["reference_bindings"],"preflight/producer differs")
    for item in p["direct_source_captures"]:
        require(bind(item["maintained"]["path"])==item["maintained"] and
                bind(item["capture"]["path"])==item["capture"],"direct source/capture changed")
    return a,refs,p


def launch():
    checked();require(not (OUT/"outer.json").exists() and not (OUT/"launch.json").exists(),
                      "attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-B","-m","probes.training_set_completion.recurrence_history_read_stage.run","run"]
    begun=time.monotonic()
    with (OUT/"stdout.log").open("x") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        code=child.wait()
    packet={"command":command,"child_pid":child.pid,
            "outer_seconds":time.monotonic()-begun,"returncode":code,"terminal":True}
    write_new(OUT/"outer.json",packet)
    print(json.dumps(packet))
    require(code==0,"GPU child failed; no retry")


def run():
    a,refs,p=checked();require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
                              "attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();device=torch.device("cuda:0");handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"emitted_target_tokens":0,"reused_calls":0}
    receipt={"status":"running","pid":os.getpid(),"begun_unix":time.time(),
             "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),
             "repair":bind(REPAIR),"device_qualification":bind(QUALIFICATION),
             "prior_charged_sequence_gpu_hours":p["prior_charged_sequence_gpu_hours"],
             "producer":bind(Path(__file__)),"counts":counts,"arms":[]}
    write_new(OUT/"launch.json",receipt)
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        q,identity=base.load_model("untied",device)
        expected=p["source_identity"]
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in expected.items() if k!="loader_source"} and
                all(identity["loader_source"][k]==expected["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"]==a["loader_crosswalk"]["maintained"]["path"],
                "effective checkpoint/loader differs")
        model=q.model.eval();batch,raw,trace,sr,planning=source(q,a,device)
        require(sr["input_identity"]==json.loads(Path(a["source_bindings"]["runtime_receipt"]["path"]).read_text())["input_identity"] and
                int(q.tokenizer.pad_token_id)==p["pad_id"] and
                sorted(q.tokenizer.all_special_ids)==p["special_ids"],"GPU source/tokenizer differs")
        pad=p["pad_id"];special=frozenset(p["special_ids"])
        layers=list(model.model.language_model.layers);attentions=[x.self_attn for x in layers]
        require(len(layers)==len(attentions)==28 and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual 28 text layers changed")
        active={}
        def top(_m,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=118,"model-forward cap")
            active["actual_input"]=input_hashes(kwargs,kwargs["attention_mask"])
            require(active["actual_input"]==active["expected_input_hashes"],"top-level full input changed")
        def vision(_m,_args):
            counts["vision_forwards"]+=1;require(counts["vision_forwards"]<=118,"vision-forward cap")
        handles += [model.register_forward_pre_hook(top,with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(vision)]
        for i,x in enumerate(attentions):
            def hook(_m,_args,kwargs,layer=i):
                mask=kwargs.get("attention_mask")
                require(isinstance(mask,torch.Tensor) and mask.ndim==4 and
                        torch.equal(mask,active["expected_mask"]),f"layer {layer} consumed wrong mask")
                active["layers"].append(layer);active["layer_mask_hashes"].append(base.tensor_hash(mask))
            handles.append(x.register_forward_pre_hook(hook,with_kwargs=True))
        for i,x in enumerate(layers):
            def state_hook(_m,_args,output,layer=i):
                value=output[0] if isinstance(output,tuple) else output
                require(isinstance(value,torch.Tensor) and value.ndim==3,"layer states unavailable")
                active["history"].append(value[TARGET,PROMPT:PROMPT+OFFSET].detach().cpu().float())
                active["companions"].append(value[[0,1,3],-1].detach().cpu().float())
            handles.append(x.register_forward_hook(state_hook))
        receipt["effective_identity"]=identity
        baselines={}
        with torch.inference_mode():
            for arm in ARMS:
                emitted=[];record={"arm":arm,"history":spec(a,arm)["history"],
                                   "mask_mode":spec(a,arm)["mask_mode"],"steps":[],"stop":None}
                receipt["arms"].append(record)
                if arm=="header_A":
                    require(len(receipt["arms"])==7 and
                            all(x["stop"]=="complete" and len(x["steps"])==9
                                for x in receipt["arms"][:6]),"six known arms not qualified")
                for t in range(spec(a,arm)["max_tokens"]):
                    full=step_inputs(model,batch,raw,pad,a,arm,emitted,
                                     previous=emitted[-1] if t else None)
                    mask,native,selected=mask_for(full,a,arm)
                    actual=native if spec(a,arm)["mask_mode"]=="native" else mask
                    active.update(full=full,expected_mask=actual,
                                  expected_input_hashes=input_hashes(full,mask))
                    out=caller(model,full,a,arm,active)
                    torch.cuda.synchronize(device)
                    require(len(active["layers"])==len(active["history"])==len(active["companions"])==28 and
                            active.get("actual_input") is not None,"actual consumers/states incomplete")
                    logits=out.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape==(4,152670) and torch.isfinite(logits).all().item(),
                            "invalid full-vocabulary vectors")
                    chosen=int(torch.argmax(logits[TARGET]).item())
                    verify_online(model,batch,raw,pad,a,arm,emitted,full,active,logits,chosen)
                    payload={"arm":arm,"step":t,"logits":logits,
                             "actual_mask":actual.detach().cpu(),
                             "historical_by_layer":torch.stack(active["history"]),
                             "companions_by_layer":torch.stack(active["companions"])}
                    rawpath=OUT/f"{arm}-step{t}.pt";torch.save(payload,rawpath)
                    inputpath=OUT/f"inputs-{arm}-step{t}.json"
                    write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                    entry=entry_for(full,mask,native,selected,active,chosen,t,rawpath,inputpath)
                    record["steps"].append(entry)
                    actual_serialized={k:torch.tensor(v,dtype=torch.long) for k,v in
                        json.loads(inputpath.read_text()).items()}
                    verify_serialized(model,batch,raw,pad,a,arm,emitted,entry,actual_serialized,payload)
                    entry["accepted_reference_max_error"]=reference(a,refs,arm,t,full,logits,chosen)
                    compare_states(a,arm,t,payload,logits,baselines)
                    entry["source_trace_parity"]=trace_check(logits,trace,raw,arm,t)
                    emitted.append(chosen);counts["emitted_target_tokens"]+=1
                    require(counts["emitted_target_tokens"]<=118,"emitted-token cap")
                    parsed=parse_row(emitted,special)
                    record["emitted"]=list(emitted);record["stop"]=parsed["stop"]
                    receipt["counts"]=dict(counts)
                    write_new(OUT/f"checkpoint-{arm}-{t}.json",receipt)
                    if parsed["stop"] is not None:break
                require(record["stop"] is not None,"trajectory did not terminate at cap")
                if arm in ARMS[:6]:
                    expected=refs["accepted_references"][spec(a,arm)["reference_key"]]["emitted"]
                    require(record["stop"]=="complete" and record["emitted"]==expected and
                            len(record["steps"])==9,"known nine-step arm failed qualification")
                if arm.startswith("native_"):
                    baselines[arm]=[torch.load(OUT/f"{arm}-step{t}.pt",map_location="cpu",weights_only=True)
                                    for t in range(9)]
        receipt["status"]="candidate_raw_complete"
    except BaseException as exc:
        receipt["status"]="technical_invalid"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    finally:
        for h in handles:h.remove()
        if torch.cuda.is_available():torch.cuda.synchronize(device)
        receipt["counts"]=dict(counts);receipt["internal_seconds"]=time.monotonic()-started
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["artifact_bytes"]=sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"]=os.getpid();write_new(OUT/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"counts":counts,
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_raw_complete","technical failure; no retry")


def classify(row,a):
    if row["stop"]!="complete":return row["stop"]
    if row["description_ids"]!=[8987]:return "other_class"
    if row["geometry"]!="valid":return "invalid_geometry"
    ia=iou_xyxy(row["box"],a["region_rule"]["A"]);iff=iou_xyxy(row["box"],a["region_rule"]["F"])
    row["iou_A"]=ia;row["iou_F"]=iff
    guard=a["region_rule"]["numerical_guard"]
    if min(abs(ia-.5),abs(iff-.1),abs(iff-.5),abs(ia-.1))<=guard:return "numerical_HOLD"
    if ia>=.5 and iff<=.1:return "broad_A"
    if iff>=.5 and ia<=.1:return "fragment_F"
    return "neither_region"


def readback():
    a,refs,p=checked();r=json.loads((OUT/"receipt.json").read_text())
    validate_arm_order(r["arms"])
    require(r["status"]=="candidate_raw_complete" and r["admission"]==bind(ADMISSION) and
            r["repair"]==bind(REPAIR) and
            r["device_qualification"]==bind(QUALIFICATION) and
            r["prior_charged_sequence_gpu_hours"]==p["prior_charged_sequence_gpu_hours"] and
            r["counts"]["reused_calls"]==0 and
            not (OUT/"readback.json").exists(),"terminal receipt/arms/readback changed")
    outer=json.loads((OUT/"outer.json").read_text())
    require(outer["terminal"] and outer["returncode"]==0 and outer["outer_seconds"]>0,
            "outer process not terminal")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold readback loaded language model")
    batch,raw,trace,sr,planning=source(q,a,torch.device("cpu"))
    require(int(q.tokenizer.pad_token_id)==p["pad_id"] and
            sorted(q.tokenizer.all_special_ids)==p["special_ids"],"cold tokenizer/source changed")
    pad=p["pad_id"];special=frozenset(p["special_ids"]);fixture=base.ConfigOnlyRope()
    result={"status":"candidate_cold_readback_passed","admission":bind(ADMISSION),
            "repair":bind(REPAIR),"device_qualification":bind(QUALIFICATION),
            "receipt":bind(OUT/"receipt.json"),"outer":bind(OUT/"outer.json"),
            "arms":[],"counts":r["counts"],"outer_seconds":outer["outer_seconds"],
            "cumulative_sequence_gpu_hours":p["prior_charged_sequence_gpu_hours"]+
                                             outer["outer_seconds"]/3600}
    baselines={};count=0
    for record in r["arms"]:
        arm=record["arm"];emitted=[];summaries=[];native_payloads=[]
        require(record["history"]==spec(a,arm)["history"] and
                record["mask_mode"]==spec(a,arm)["mask_mode"],"cold arm mode/history changed")
        for t,e in enumerate(record["steps"]):
            require(e["step"]==t and bind(e["raw"]["path"])==e["raw"] and
                    bind(e["inputs"]["path"])==e["inputs"],"cold raw/input binding changed")
            stored={k:torch.tensor(v,dtype=torch.long) for k,v in
                    json.loads(Path(e["inputs"]["path"]).read_text()).items()}
            payload=torch.load(e["raw"]["path"],map_location="cpu",weights_only=True)
            full=verify_serialized(fixture,batch,raw,pad,a,arm,emitted,e,stored,payload)
            logits=payload["logits"]
            require(payload["historical_by_layer"].shape[0]==
                    payload["companions_by_layer"].shape[0]==28,
                    "cold states incomplete")
            compare_states(a,arm,t,payload,logits,baselines)
            reference(a,refs,arm,t,full,logits,e["chosen"])
            trace_check(logits,trace,raw,arm,t)
            v=logits[TARGET].double();top=torch.topk(v,2)
            summaries.append({"step":t,"chosen":e["chosen"],"top2_ids":top.indices.tolist(),
                              "top2_logits":top.values.tolist(),
                              "log_normalizer":float(torch.logsumexp(v,-1)),"raw":e["raw"]})
            emitted.append(e["chosen"]);count+=1
            stop=parse_row(emitted,special)["stop"]
            require((stop is None) if t<len(record["steps"])-1 else stop==record["stop"],
                    "cold parser/stop changed")
            if arm.startswith("native_"):native_payloads.append(payload)
        require(emitted==record["emitted"] and len(emitted)<=spec(a,arm)["max_tokens"],
                "cold own prefix/count changed")
        parsed=parse_row(emitted,special)
        desc=q.tokenizer.decode(parsed["description_ids"],skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        box=[x-151670 for x in parsed["box_ids"]] if len(parsed["box_ids"])==4 else None
        geometry=("valid" if box[0]<box[2] and box[1]<box[3] else "invalid") if box else "no_complete_box"
        row={"arm":arm,"emitted":emitted,"stop":record["stop"],
             "description_ids":parsed["description_ids"],"description":desc,
             "box":box,"geometry":geometry,"steps":summaries}
        row["region"]=classify(row,a)
        result["arms"].append(row)
        if arm.startswith("native_"):baselines[arm]=native_payloads
        if arm in ARMS[:6]:
            expected=refs["accepted_references"][spec(a,arm)["reference_key"]]["emitted"]
            require(record["stop"]=="complete" and emitted==expected and len(emitted)==9,
                    "cold known reference qualification failed")
    require(count==r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==
            r["counts"]["emitted_target_tokens"]<=118 and len(result["arms"])==10,
            "cold finite calls/counts changed")
    observed={row["arm"]:row["region"] for row in result["arms"] if row["arm"] in ARMS[6:]}
    result["partial_regions"]=observed
    result["primary_pass"]=all(observed.get(k)==v for k,v in a["primary"].items()
                              if k in ARMS[6:])
    result["comparator_pass"]=all(observed.get(k)==v for k,v in a["comparator"].items()
                                 if k in ARMS[6:])
    if any(v=="numerical_HOLD" for v in observed.values()):result["shared_decision"]="numerical_HOLD"
    elif any(observed.get(k)!="broad_A" for k in ("header_F","coordinate_F")):
        result["shared_decision"]="control_changed"
    elif result["primary_pass"]:result["shared_decision"]="header_edge_signature"
    elif result["comparator_pass"]:result["shared_decision"]="coordinate_edge_signature"
    elif observed["header_A"]==observed["coordinate_A"]=="broad_A":
        result["shared_decision"]="both_single_blocks_change"
    elif observed["header_A"]==observed["coordinate_A"]=="fragment_F":
        result["shared_decision"]="neither_single_block_changes"
    else:result["shared_decision"]="mixed_or_other"
    image=Image.open(a["source_bindings"]["image"]["path"]).convert("RGB");draw=ImageDraw.Draw(image)
    for label,box,color in [("A",a["region_rule"]["A"],"#00ee77"),
                            ("F",a["region_rule"]["F"],"#ff5533")]+[
                            (x["arm"],x["box"],col) for x,col in zip(result["arms"][6:],
                            ("#00d9ff","#ee00dd","#ffff00","#ffffff"),strict=True)]:
        if box is None or box[0]>=box[2] or box[1]>=box[3]:continue
        xy=[round(v*(image.width if k%2==0 else image.height)/1000) for k,v in enumerate(box)]
        draw.rectangle(xy,outline=color,width=4);draw.text((xy[0],max(0,xy[1]-14)),label,fill=color)
    image.save(OUT/"overlay.png");result["overlay"]=bind(OUT/"overlay.png")
    write_new(OUT/"readback.json",result)
    print(json.dumps({"status":result["status"],"regions":observed,
                      "decision":result["shared_decision"],"counts":r["counts"]}))


def main():
    arg=argparse.ArgumentParser();arg.add_argument("action",choices=("preflight","launch","run","readback"))
    {"preflight":preflight,"launch":launch,"run":run,"readback":readback}[arg.parse_args().action]()


if __name__=="__main__":main()
