"""Three free-row full-prefix arms at train351017's first localization collapse."""
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
from PIL import Image, ImageDraw
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_book_first_revisit import run as base
from probes.training_set_completion.recurrence_free_header_routing.run import parse_row
from src.data.geometry import iou_xyxy


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-first-localization-collapse"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-localization-collapse/attempt-001")
SHAS = {PROTOCOL: "3ea786e900a491b32c540f997456ebbfdcc69ed9b123de9e1f94ceca6652c708",
        ADMISSION: "2736fe8e963b2d3aa134157457d550074cc43c4659b34451c7bbce38b8909954"}
ARMS = ("native", "independent_identity_mask_sham", "latest_record_read_mask")
TARGET, OFFSET, PROMPT, TOL = 2, 9, 1362, 2e-4
ROW0 = (151646, 8987, 151647, 151648, 151670, 151683, 152206, 152669, 151649)
ROW1 = (151646, 8987, 151647, 151648, 151670, 151670, 151703, 151756, 151649)
KEYS = ("input_ids", "attention_mask", "position_ids", "cache_position")
require, bind, write_new = base.require, base.bind, base.write_new


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-finite-first-localization-collapse" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["source"]["group"] == "refined-03" and a["source"]["target_index"] == TARGET and
            a["source"]["history_raw_end"] == OFFSET and a["source"]["prior_keys_raw"] == [0, 9] and
            a["source"]["source_row0_tokens"] == list(ROW0) and
            a["source"]["source_row1_tokens"] == list(ROW1) and
            a["source"]["A_box"] == [0, 13, 536, 999] and
            a["source"]["F_box"] == [0, 0, 33, 86] and
            a["arms"] == list(ARMS) and a["owned_paths"][-1] == str(OUT) and
            a["max_model_forwards"] == a["max_vision_forwards"] == a["max_emitted_tokens"] == 34 and
            a["reused_calls"] == 0 and a["artifact_planning_bytes"] == 1024**3,
            "finite admission/endpoint changed")
    for name in ("protocol", "visual_acceptance", "visual_source_record", "source_panel"):
        b = a[name]
        require(bind(b["path"]) == b, f"admission dependency changed: {name}")
    for name in ("raw", "trace", "runtime_receipt", "image"):
        b = a["source_bindings"][name]
        require(bind(b["path"]) == b, f"original source changed: {name}")
    cross = a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"]) == cross["maintained"] and
            cross["maintained"]["sha256"] == cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"]) == cross["prior_acceptance"],
            "maintained loader crosswalk changed")
    return a


def source(q, a, device):
    b = a["source_bindings"]
    source_rows = json.loads(Path(b["raw"]["path"]).read_text())["rows"]
    boundary = {"group": "refined-03", "batch_index": TARGET, "image_id": 351017,
                "raw_path": b["raw"]["path"], "trace_path": b["trace"]["path"],
                "receipt_path": b["runtime_receipt"]["path"],
                "native_tokens": source_rows[TARGET]["token_ids"],
                "native_token_hash": base.token_hash(source_rows[TARGET]["token_ids"])}
    panel = json.loads(Path(a["source_panel"]["path"]).read_text())
    batch, raw, trace, group, planning = base._source(boundary, "untied", panel, q, device)
    receipt = json.loads(Path(b["runtime_receipt"]["path"]).read_text())
    require(len(raw) == len(group["cases"]) == 4 and
            list(batch.request_ids) == a["source_bindings"]["input_identity_summary"]["request_ids"] and
            list(batch.request_ids) == ["coco2017_val_000000016228", "coco2017_train_000000007116",
                                        "coco2017_train_000000351017", "coco2017_train_000000417044"] and
            [len(x["token_ids"]) for x in raw] == [255, 37, 3084, 3084] and
            list(map(len, batch.prompt_token_ids)) == [1336, 1362, 1362, 1320] and
            int(batch.inputs["pixel_values"].numel()) == 24502272 and
            base.input_identity(batch) == receipt["input_identity"] and
            raw[TARGET]["token_ids"][:OFFSET] == list(ROW0) and
            raw[TARGET]["token_ids"][OFFSET:OFFSET+9] == list(ROW1) and
            receipt["identity"]["adapter"]["adapter_path"] == b["adapter_path"] and
            receipt["identity"]["embedding"]["identity"]["delta_path"] == b["embedding_delta_path"],
            "original four-request source/identity/row changed")
    return batch, raw, trace, receipt, planning


def step_inputs(model, batch, raw, pad, arm, emitted, *, previous=None, target=TARGET, prefix=OFFSET):
    t = len(emitted)
    require(arm in ARMS and target == TARGET and prefix == OFFSET and
            0 <= t < (16 if arm == ARMS[2] else 9) and
            (t == 0 or emitted[-1] == previous), "wrong target/prefix/own greedy token/step")
    tails = base._prefix_tokens(raw, OFFSET+t, pad)
    tails[TARGET] = list(ROW0) + list(emitted)
    histories = [list(p)+tail for p,tail in zip(batch.prompt_token_ids,tails,strict=True)]
    full = base.exact_history_inputs(model,batch.inputs,histories,pad_token_id=pad,logits_to_keep=1)
    width = int(full["input_ids"].shape[1])
    full["cache_position"] = torch.arange(width,device=full["input_ids"].device)
    require(width == PROMPT+OFFSET+t and full["attention_mask"].shape == (4,width) and
            full["position_ids"].shape == (3,4,width) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            tails[TARGET][:OFFSET] == list(ROW0) and tails[TARGET][OFFSET:] == list(emitted) and
            all(tails[i] == raw[i]["token_ids"][:OFFSET+t] for i in (0,1,3)) and
            full["attention_mask"][:,PROMPT:].all().item() and
            all(int(full["attention_mask"][i,:PROMPT].sum()) == len(batch.prompt_token_ids[i]) for i in range(4)),
            "source companions/row0/positions/own prefix changed")
    return full


def mask_for(full, arm, *, target=TARGET, query=None, key=None, source_hash=None):
    width = int(full["input_ids"].shape[1]); t = width-PROMPT-OFFSET
    qspan = (PROMPT+OFFSET,PROMPT+OFFSET+t); kspan = (PROMPT,PROMPT+OFFSET)
    require(arm in ARMS and target == TARGET and 0 <= t <= 15 and
            query in (None,qspan) and key in (None,kspan) and
            (source_hash is None or base.tensor_hash(full["attention_mask"]) == source_hash),
            "wrong target/query/key/source mask")
    native = base.native_4d(full["attention_mask"])
    selected = torch.zeros_like(native)
    selected[TARGET,0,qspan[0]:qspan[1],kspan[0]:kspan[1]] = True
    require(int(selected.sum()) == 9*t and (t == 0 or bool(native[selected].all())),
            "selected history rectangle was not native readable")
    actual = native.clone()
    if arm == ARMS[2]: actual[selected] = False
    verify_rectangle(native,actual,selected,arm,t)
    return (full["attention_mask"] if arm == ARMS[0] else actual),native,selected


def verify_rectangle(native, actual, selected, arm, t):
    require(native.shape == actual.shape == selected.shape and int(selected.sum()) == 9*t and
            torch.equal(actual[~selected],native[~selected]) and
            torch.equal(actual[selected],torch.zeros_like(native[selected]) if arm == ARMS[2] else native[selected]),
            "selected/complement mask changed")


def caller(model, full, arm, observer):
    mask,native,selected=mask_for(full,arm,source_hash=base.tensor_hash(full["attention_mask"]))
    observer.update(expected_mask=native if arm == ARMS[0] else mask,native_mask=native,
                    selected=selected,actual_input=None,layers=[],layer_mask_hashes=[],history=[],companions=[])
    return model(**{**full,"attention_mask":mask})


def verify_step(model,batch,raw,pad,arm,emitted,full,seen,logits,chosen,*,previous=None):
    expected=step_inputs(model,batch,raw,pad,arm,emitted,previous=previous)
    mask,native,selected=mask_for(expected,arm)
    hashes={k:base.tensor_hash(expected[k]) for k in KEYS}
    hashes["attention_mask"] = base.tensor_hash(mask)
    require(all(torch.equal(full[k],expected[k]) for k in KEYS) and seen["actual_input"] == hashes and
            seen["layers"] == list(range(28)) and
            seen["layer_mask_hashes"] == [base.tensor_hash(seen["expected_mask"])]*28 and
            torch.equal(seen["expected_mask"],native if arm == ARMS[0] else mask) and
            torch.equal(seen["selected"],selected) and
            chosen == int(torch.argmax(logits[TARGET]).item()),
            "actual source/input/28-layer mask/greedy differs")
    return hashes


def cpu_checks(batch,raw,pad,special):
    model=base.ConfigOnlyRope(); checks=[]
    require(parse_row(list(ROW0),special)["stop"] == "complete" and
            parse_row(list(ROW1),special)["stop"] == "complete" and
            parse_row([151645],special)["stop"] == "eos" and
            parse_row([151649],special)["stop"] == "early_row_terminator" and
            parse_row([151646,8987,151647,151648,152670],special)["stop"] == "malformed_coordinate" and
            parse_row([151646,8987,151647,151648,152669],special)["stop"] is None and
            parse_row([151646,8987,151647,151648,151670,151671,151672,151673,151670],special)["stop"] == "malformed_terminator" and
            parse_row([151646]+[8987]*15,special)["stop"] == "cap", "parser boundary changed")
    checks.append("complete_eos_early_malformed_cap_exclusive")
    cfg=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    cfg._attn_implementation="sdpa"
    for t in (0,1,15):
        emitted=list(ROW1[:t]) if t<=9 else [151646]+[8987]*(t-1)
        full=step_inputs(model,batch,raw,pad,ARMS[2],emitted,previous=emitted[-1] if t else None)
        reference=create_causal_mask(cfg,torch.empty((*full["attention_mask"].shape,1)),
            full["attention_mask"],full["cache_position"],None,position_ids=full["position_ids"][0])
        require(torch.equal(base.native_4d(full["attention_mask"]),reference),"installed SDPA causal mask changed")
        for arm in (ARMS if t < 9 else (ARMS[2],)):
            class Fake:
                def __call__(self,**kwargs):
                    seen["actual_input"]={k:base.tensor_hash(kwargs[k]) for k in KEYS}
                    seen["actual_input"]["attention_mask"]=base.tensor_hash(kwargs["attention_mask"])
                    z=torch.zeros((4,1,152670));z[TARGET,0,151646]=1
                    return SimpleNamespace(logits=z)
            seen={}; logits=caller(Fake(),full,arm,seen).logits[:,-1]
            seen["layers"]=list(range(28));seen["layer_mask_hashes"]=[base.tensor_hash(seen["expected_mask"])]*28
            verify_step(model,batch,raw,pad,arm,emitted,full,seen,logits,151646,previous=emitted[-1] if t else None)
            checks.append(f"actual_caller_receipt_{arm}_t{t}")
            if t == 0 and arm == ARMS[2]:
                require(torch.equal(mask_for(full,arm)[0],mask_for(full,ARMS[1])[0]),"t0 mask was not native")
        for label,kw in (("target",{"target":3}),
                         ("query",{"query":(PROMPT+OFFSET-1,PROMPT+OFFSET+t)}),
                         ("key",{"key":(PROMPT+1,PROMPT+OFFSET)})):
            try: mask_for(full,ARMS[2],**kw)
            except ValueError: checks.append(f"reject_{label}_t{t}")
            else: raise AssertionError(label)
        native=mask_for(full,ARMS[2])[1];actual=mask_for(full,ARMS[2])[0];selected=mask_for(full,ARMS[2])[2]
        bad=actual.clone();bad[0,0,-1,0]=~bad[0,0,-1,0]
        try:verify_rectangle(native,bad,selected,ARMS[2],t)
        except ValueError:checks.append(f"reject_complement_t{t}")
        else:raise AssertionError("same-rectangle complement")
        bad=full["attention_mask"].clone();bad[0,0]=1-int(bad[0,0]);badfull={**full,"attention_mask":bad}
        try:mask_for(badfull,ARMS[2],source_hash=base.tensor_hash(full["attention_mask"]))
        except ValueError:checks.append(f"reject_source_mask_t{t}")
        else:raise AssertionError("source mask")
        seen={}; class_fake=type("Fake",(),{"__call__":lambda self,**kwargs:SimpleNamespace(logits=torch.zeros((4,1,152670)))})
        _=caller(class_fake(),full,ARMS[2],seen)
        seen["actual_input"]={k:base.tensor_hash(full[k]) for k in KEYS};seen["actual_input"]["attention_mask"]=base.tensor_hash(mask_for(full,ARMS[2])[0])
        seen["layers"]=list(range(28));seen["layer_mask_hashes"]=[base.tensor_hash(seen["expected_mask"])]*28
        logits=torch.zeros((4,152670));logits[TARGET,151646]=1
        for label,k,index in (("history","input_ids",(TARGET,PROMPT)),
                              ("companion","input_ids",(0,PROMPT)),
                              ("own_prefix","input_ids",(TARGET,PROMPT+OFFSET+t-1 if t else PROMPT+OFFSET-1)),
                              ("position","position_ids",(0,TARGET,PROMPT)),
                              ("source_attention","attention_mask",(0,PROMPT))):
            mutated={**full,k:full[k].clone()};mutated[k][index]+=1
            try:verify_step(model,batch,raw,pad,ARMS[2],emitted,mutated,seen,logits,151646,
                            previous=emitted[-1] if t else None)
            except ValueError:checks.append(f"reject_{label}_t{t}")
            else:raise AssertionError(label)
        try:verify_step(model,batch,raw,pad,ARMS[2],emitted,full,seen,logits,151647,
                        previous=emitted[-1] if t else None)
        except ValueError:checks.append(f"reject_wrong_greedy_t{t}")
        else:raise AssertionError("wrong greedy")
        if t:
            try:step_inputs(model,batch,raw,pad,ARMS[2],emitted,previous=-1)
            except ValueError:checks.append(f"reject_own_previous_t{t}")
            else:raise AssertionError("own previous")
        try:step_inputs(model,batch,raw,pad,ARMS[2],emitted,target=3)
        except ValueError:checks.append(f"reject_input_target_t{t}")
        else:raise AssertionError("input target")
    return checks


def preflight():
    a=contract();require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),"attempt already prepared")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded language model")
    batch,raw,trace,receipt,planning=source(q,a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id);special=frozenset(q.tokenizer.all_special_ids)
    checks=cpu_checks(batch,raw,pad,special)
    widths=[];fixture=base.ConfigOnlyRope()
    for t in range(16):
        emitted=list(ROW1[:t]) if t<=9 else [151646]+[8987]*(t-1)
        full=step_inputs(fixture,batch,raw,pad,ARMS[2],emitted,previous=emitted[-1] if t else None)
        widths.append(int(full["input_ids"].shape[1]))
    pixel_elements=int(batch.inputs["pixel_values"].numel())
    require(widths==list(range(1371,1387)) and pixel_elements==24502272 and
            len(checks)>=30,"finite source-shape/caller checks failed")
    cfg=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    vocab,hidden=int(cfg.vocab_size),int(cfg.hidden_size)
    forecast=2*81.712466598*(34/30)*(pixel_elements/24403968)*(sum(widths)/len(widths)/1377)
    artifact_forecast=34*(4*vocab*4+28*(OFFSET+3)*hidden*4+4*1386**2+500000)+32*1024**2
    require(artifact_forecast<a["artifact_planning_bytes"],"artifact forecast exceeds envelope")
    predecessor=json.loads(base.PREFLIGHT.read_text())
    from probes.training_set_completion.recurrence_free_header_routing import run as free
    free_pre=json.loads(free.PREFLIGHT.read_text())
    direct=[Path(__file__),Path(base.__file__),Path(free.__file__),
            Path(inspect.getfile(create_causal_mask)),Path(inspect.getfile(modeling_qwen3_vl)),
            Path(inspect.getfile(iou_xyxy))]
    direct += [Path(c["maintained"]["path"]) for c in predecessor["captures"]]
    direct += [Path(c["maintained"]["path"]) for c in free_pre["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(direct):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_first_localization_collapse.run"]
    packet={"status":"cpu_qualified_before_gpu","admission":bind(ADMISSION),"protocol":bind(PROTOCOL),
            "visual_acceptance":bind(a["visual_acceptance"]["path"]),"producer":bind(Path(__file__)),
            "source_bindings":a["source_bindings"],"source_identity":receipt["identity"],
            "input_identity_sha256":a["source_bindings"]["input_identity_sha256"],
            "request_ids":list(batch.request_ids),"prompt_lengths":list(map(len,batch.prompt_token_ids)),
            "raw_lengths":[len(x["token_ids"]) for x in raw],"pad_id":pad,"special_ids":sorted(special),
            "image_grids":[list(x) for x in batch.image_grids],"pixel_elements":pixel_elements,
            "widths":widths,"physical_query_t15":[PROMPT+OFFSET,PROMPT+OFFSET+15],
            "physical_key":[PROMPT,PROMPT+OFFSET],"cpu_checks":checks,
            "forecast_outer_seconds":forecast,"artifact_forecast_bytes":artifact_forecast,
            "direct_source_captures":captures,
            "commands":{"preflight":command+["preflight"],"gpu":command+["run"],"readback":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),"captures":len(captures),
                      "widths":[widths[0],widths[-1]],"pixels":pixel_elements,
                      "forecast_seconds":forecast,"artifact_bytes":artifact_forecast}))


def checked():
    a=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["admission"]==bind(ADMISSION) and
            p["protocol"]==bind(PROTOCOL) and p["producer"]==bind(Path(__file__)),
            "preflight/producer binding changed")
    for c in p["direct_source_captures"]:
        require(bind(c["maintained"]["path"])==c["maintained"] and
                bind(c["capture"]["path"])==c["capture"],"captured direct source changed")
    return a,p


def run():
    a,p=checked();require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
                          "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();device=torch.device("cuda:0");handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"emitted_target_tokens":0,"reused_calls":0}
    receipt={"status":"running","pid":os.getpid(),"begun_unix":time.time(),
             "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
             "counts":counts,"arms":[]}
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
                "effective model/loader differs")
        model=q.model.eval();batch,raw,trace,sr,planning=source(q,a,device)
        require(sr["input_identity"]==json.loads(Path(a["source_bindings"]["runtime_receipt"]["path"]).read_text())["input_identity"] and
                int(q.tokenizer.pad_token_id)==p["pad_id"] and
                sorted(q.tokenizer.all_special_ids)==p["special_ids"],"GPU source/tokenizer differs")
        pad=p["pad_id"];special=frozenset(p["special_ids"])
        layers=list(model.model.language_model.layers)
        attentions=[x.self_attn for x in layers]
        require(len(layers)==len(attentions)==28 and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual 28 text-layer route changed")
        active={}
        def top(_m,_args,kwargs):
            counts["model_forwards"]+=1;require(counts["model_forwards"]<=34,"model-forward cap")
            active["actual_input"]={k:base.tensor_hash(kwargs[k]) for k in KEYS}
            active["actual_input"]["attention_mask"]=base.tensor_hash(kwargs["attention_mask"])
            require(active["actual_input"]==active["expected_input_hashes"],"top-level actual input changed")
        def vision(_m,_args):
            counts["vision_forwards"]+=1;require(counts["vision_forwards"]<=34,"vision-forward cap")
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
        with torch.inference_mode():
            for arm in ARMS:
                emitted=[];record={"arm":arm,"steps":[],"stop":None};receipt["arms"].append(record)
                limit=16 if arm==ARMS[2] else 9
                for t in range(limit):
                    previous=emitted[-1] if t else None
                    full=step_inputs(model,batch,raw,pad,arm,emitted,previous=previous)
                    mask,native,selected=mask_for(full,arm)
                    expected_mask=native if arm==ARMS[0] else mask
                    input_hashes={k:base.tensor_hash(full[k]) for k in KEYS}
                    input_hashes["attention_mask"]=base.tensor_hash(mask)
                    active.update(expected_mask=expected_mask,expected_input_hashes=input_hashes)
                    out=caller(model,full,arm,active)
                    torch.cuda.synchronize(device)
                    require(len(active["layers"])==len(active["history"])==len(active["companions"])==28 and
                            active.get("actual_input") is not None,"actual consumers/states incomplete")
                    logits=out.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape==(4,152670) and torch.isfinite(logits).all().item(),"invalid full vectors")
                    chosen=int(torch.argmax(logits[TARGET]).item())
                    hashes=verify_step(model,batch,raw,pad,arm,emitted,full,active,logits,chosen,previous=previous)
                    payload={"arm":arm,"step":t,"logits":logits,"actual_mask":expected_mask.detach().cpu(),
                             "historical_by_layer":torch.stack(active["history"]),
                             "companions_by_layer":torch.stack(active["companions"])}
                    rawpath=OUT/f"{arm}-step{t}.pt";torch.save(payload,rawpath)
                    inputpath=OUT/f"inputs-{arm}-step{t}.json"
                    write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                    entry={"step":t,"raw":bind(rawpath),"inputs":bind(inputpath),"input_hashes":hashes,
                           "actual_layers":active["layers"],"actual_layer_mask_hashes":active["layer_mask_hashes"],
                           "selected_cells":int(selected.sum()),"selected_native_hash":base.tensor_hash(native[selected]),
                           "selected_actual_hash":base.tensor_hash(expected_mask[selected]),
                           "complement_native_hash":base.tensor_hash(native[~selected]),
                           "complement_actual_hash":base.tensor_hash(expected_mask[~selected]),
                           "actual_mask_hash":base.tensor_hash(expected_mask),"chosen":chosen,
                           "internal_seconds":time.monotonic()-started}
                    record["steps"].append(entry)
                    if arm==ARMS[0]:
                        entry["source_trace_parity"]=[base._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],
                            role="collapse_native") for i in range(4)]
                        require(all(x["passed"] for x in entry["source_trace_parity"]) and
                                chosen==ROW1[t],"native source chosen/top2/normalizer parity failed")
                    else:
                        entry["companion_trace_parity"]=[base._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],
                            role="collapse_companion") for i in (0,1,3)]
                        require(all(x["passed"] for x in entry["companion_trace_parity"]),
                                "active companion source parity failed")
                        if t<9:
                            old=torch.load(OUT/f"native-step{t}.pt",map_location="cpu",weights_only=True)
                            entry["history_error"]=float((payload["historical_by_layer"]-old["historical_by_layer"]).abs().max())
                            entry["companion_state_error"]=float((payload["companions_by_layer"]-old["companions_by_layer"]).abs().max())
                            entry["companion_vector_error"]=max(float((logits[i]-old["logits"][i]).abs().max()) for i in (0,1,3))
                            require(max(entry["history_error"],entry["companion_state_error"],entry["companion_vector_error"])<=TOL,
                                    "historical/companion state or vectors changed")
                            if arm==ARMS[1]:
                                entry["full_vector_error"]=float((logits-old["logits"]).abs().max())
                                require(entry["full_vector_error"]<=TOL and chosen==ROW1[t],
                                        "independent sham vector/greedy differs")
                            if arm==ARMS[2] and t==0:
                                entry["t0_full_vector_error"]=float((logits-old["logits"]).abs().max())
                                require(entry["t0_full_vector_error"]<=TOL and chosen==ROW1[0],
                                        "t0 native agreement failed")
                        else:
                            entry["matched_native_full_vector"]="UNAVAILABLE_not_executed"
                    emitted.append(chosen);counts["emitted_target_tokens"]+=1
                    require(counts["emitted_target_tokens"]<=34,"emitted-token cap")
                    parsed=parse_row(emitted,special)
                    record["emitted"]=list(emitted);record["stop"]=parsed["stop"]
                    receipt["counts"]=dict(counts)
                    write_new(OUT/f"checkpoint-{arm}-{t}.json",receipt)
                    if parsed["stop"] is not None:break
                require(record["stop"] is not None,"trajectory did not stop by cap")
                if arm in ARMS[:2]:
                    require(record["stop"]=="complete" and record["emitted"]==list(ROW1),
                            "native/sham F qualification failed")
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
        receipt["artifact_bytes"]=sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"]=os.getpid()
        write_new(OUT/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"counts":counts,
                      "internal_seconds":receipt["internal_seconds"],
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_complete","technical failure; no retry")


def readback():
    a,p=checked();r=json.loads((OUT/"receipt.json").read_text())
    require(r["status"]=="candidate_complete" and r["admission"]==bind(ADMISSION) and
            [x["arm"] for x in r["arms"]]==list(ARMS) and
            r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==r["counts"]["emitted_target_tokens"]<=34 and
            r["counts"]["reused_calls"]==0 and not (OUT/"readback.json").exists(),
            "terminal receipt/counts/readback changed")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU readback loaded language model")
    batch,raw,trace,sr,planning=source(q,a,torch.device("cpu"))
    require(int(q.tokenizer.pad_token_id)==p["pad_id"] and
            sorted(q.tokenizer.all_special_ids)==p["special_ids"],"cold tokenizer differs")
    pad=p["pad_id"];special=frozenset(p["special_ids"]);fixture=base.ConfigOnlyRope()
    result={"status":"candidate_cold_readback_passed","receipt":bind(OUT/"receipt.json"),
            "admission":bind(ADMISSION),"arms":[],"counts":r["counts"]}
    native=[];count=0
    for record in r["arms"]:
        arm=record["arm"];emitted=[];step_summaries=[]
        for t,e in enumerate(record["steps"]):
            require(e["step"]==t and e["actual_layers"]==list(range(28)) and
                    bind(e["raw"]["path"])==e["raw"] and bind(e["inputs"]["path"])==e["inputs"],
                    "cold raw/input/layer binding changed")
            full={k:torch.tensor(v,dtype=torch.long) for k,v in
                  json.loads(Path(e["inputs"]["path"]).read_text()).items()}
            payload=torch.load(e["raw"]["path"],map_location="cpu",weights_only=True)
            logits=payload["logits"]
            require(payload["arm"]==arm and payload["step"]==t and
                    logits.shape==(4,152670) and torch.isfinite(logits).all().item() and
                    payload["historical_by_layer"].shape[0]==payload["companions_by_layer"].shape[0]==28,
                    "cold full vector/states incomplete")
            expected=step_inputs(fixture,batch,raw,pad,arm,emitted,previous=emitted[-1] if t else None)
            mask,nativemask,selected=mask_for(expected,arm)
            require(all(torch.equal(full[k],expected[k]) for k in KEYS) and
                    torch.equal(payload["actual_mask"],(nativemask if arm==ARMS[0] else mask)) and
                    e["actual_mask_hash"]==base.tensor_hash(payload["actual_mask"]) and
                    e["selected_cells"]==9*t and
                    e["selected_native_hash"]==base.tensor_hash(nativemask[selected]) and
                    e["selected_actual_hash"]==base.tensor_hash(payload["actual_mask"][selected]) and
                    e["complement_native_hash"]==e["complement_actual_hash"]==base.tensor_hash(nativemask[~selected]) and
                    e["actual_layer_mask_hashes"]==[e["actual_mask_hash"]]*28,
                    "cold mask/source/consumer rectangle changed")
            seen={"actual_input":e["input_hashes"],"layers":e["actual_layers"],
                  "layer_mask_hashes":e["actual_layer_mask_hashes"],
                  "expected_mask":nativemask if arm==ARMS[0] else mask,"selected":selected}
            verify_step(fixture,batch,raw,pad,arm,emitted,full,seen,logits,e["chosen"],
                        previous=emitted[-1] if t else None)
            if arm==ARMS[0]:
                parity=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],role="cold_collapse_native")
                    for i in range(4)]
                require(all(x["passed"] for x in parity) and e["chosen"]==ROW1[t],"cold native trace failed")
                native.append(payload)
            else:
                parity=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],role="cold_collapse_companion")
                    for i in (0,1,3)]
                require(all(x["passed"] for x in parity),"cold companion trace failed")
                if t<9:
                    old=native[t]
                    require(max(float((payload["historical_by_layer"]-old["historical_by_layer"]).abs().max()),
                                float((payload["companions_by_layer"]-old["companions_by_layer"]).abs().max()),
                                max(float((logits[i]-old["logits"][i]).abs().max()) for i in (0,1,3)))<=TOL,
                            "cold historical/companion state differs")
                    if arm==ARMS[1]:
                        require(float((logits-old["logits"]).abs().max())<=TOL and e["chosen"]==ROW1[t],
                                "cold sham differs")
                    if arm==ARMS[2] and t==0:
                        require(float((logits-old["logits"]).abs().max())<=TOL and e["chosen"]==ROW1[0],
                                "cold t0 differs")
            v=logits[TARGET].double();top=torch.topk(v,2)
            step_summaries.append({"step":t,"chosen":e["chosen"],"top2_ids":top.indices.tolist(),
                "top2_logits":top.values.tolist(),"log_normalizer":float(torch.logsumexp(v,-1)),
                "chosen_logprob":float(torch.log_softmax(v,-1)[e["chosen"]]),"raw":e["raw"]})
            emitted.append(e["chosen"]);count+=1
            stop=parse_row(emitted,special)["stop"]
            require(stop is None if t<len(record["steps"])-1 else stop==record["stop"],
                    "cold parser/terminal changed")
        require(emitted==record["emitted"],"cold own generated prefix changed")
        parsed=parse_row(emitted,special)
        desc=q.tokenizer.decode(parsed["description_ids"],skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        box=[x-151670 for x in parsed["box_ids"]] if len(parsed["box_ids"])==4 else None
        geometry=("valid" if box[0]<box[2] and box[1]<box[3] else "invalid") if box else "no_complete_box"
        result["arms"].append({"arm":arm,"emitted":emitted,"stop":record["stop"],
            "description_ids":parsed["description_ids"],"description":desc,"box":box,
            "geometry":geometry,"steps":step_summaries})
    require(count==r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==r["counts"]["emitted_target_tokens"] and
            count<=34 and len(result["arms"])==3,"cold finite calls/arms changed")
    treatment=result["arms"][2]
    a_box=a["source"]["A_box"];f_box=a["source"]["F_box"]
    if treatment["stop"]!="complete":outcome=treatment["stop"]
    elif treatment["description_ids"]!=[8987]:outcome="other_class"
    elif treatment["geometry"]!="valid":outcome="invalid_geometry"
    else:
        ia=iou_xyxy(treatment["box"],a_box);iff=iou_xyxy(treatment["box"],f_box)
        if min(abs(ia-.5),abs(iff-.1))<=1e-6:outcome="numerical_HOLD"
        elif ia>=.5 and iff<=.1:outcome="broad_A"
        elif iff>=.5 and ia<=.1:outcome="fragment_F"
        else:outcome="neither_region"
        treatment["iou_A"]=ia;treatment["iou_F"]=iff
    result["primary_outcome"]=outcome
    result["primary_pass"]=outcome=="broad_A"
    result["secondary_exact_row0"]=(treatment["stop"]=="complete" and treatment["emitted"]==list(ROW0))
    result["first_divergence_from_row0"]=next((i for i,(x,y) in enumerate(zip(treatment["emitted"],ROW0)) if x!=y),
                                               min(len(treatment["emitted"]),len(ROW0)))
    image=Image.open(a["source_bindings"]["image"]["path"]).convert("RGB");draw=ImageDraw.Draw(image)
    for label,box,color in (("A row0",a_box,"#00ee77"),("F row1",f_box,"#ff5533"),
                            ("masked",treatment["box"],"#00d9ff")):
        if box is None:continue
        xy=[round(v*(image.width if k%2==0 else image.height)/1000) for k,v in enumerate(box)]
        draw.rectangle(xy,outline=color,width=4);draw.text((xy[0],max(0,xy[1]-14)),label,fill=color)
    image.save(OUT/"overlay.png")
    result["overlay"]=bind(OUT/"overlay.png")
    result["outer"]=bind(OUT/"outer.json")
    write_new(OUT/"readback.json",result)
    print(json.dumps({"status":result["status"],"rows":{x["arm"]:x["emitted"] for x in result["arms"]},
                      "primary":outcome,"secondary_exact":result["secondary_exact_row0"],"counts":r["counts"]}))


def main():
    arg=argparse.ArgumentParser();arg.add_argument("action",choices=("preflight","run","readback"))
    {"preflight":preflight,"run":run,"readback":readback}[arg.parse_args().action]()


if __name__=="__main__":main()
