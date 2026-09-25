"""Fixed written A/F history crossed with free-row read access at train351017."""
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
UNIT = REPO / "research/experiments/2026-09-24-recurrence-collapse-history-content"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-collapse-history-content/attempt-001")
SHAS = {PROTOCOL: "38afdc7ccca4a74997b6833faa17022a2d1bd1f5bc21bc94c383fd62df7ef0de",
        ADMISSION: "66bb641d9c04a233e9f91566835b4ea123aaf6b77fd727aeb7b17194fb5f106b"}
ARMS = ("native_A", "identity_write_A", "masked_A", "native_F", "masked_F")
MASKED = (ARMS[2], ARMS[4])
F_ARMS = (ARMS[3], ARMS[4])
TARGET, OFFSET, PROMPT, TOL = 2, 9, 1362, 2e-4
ROW0 = (151646, 8987, 151647, 151648, 151670, 151683, 152206, 152669, 151649)
ROW1 = (151646, 8987, 151647, 151648, 151670, 151670, 151703, 151756, 151649)
KEYS = ("input_ids", "attention_mask", "position_ids", "cache_position")
require, bind, write_new = base.require, base.bind, base.write_new


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-finite-written-extent-by-read-access" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["source"]["group"] == "refined-03" and a["source"]["target_index"] == TARGET and
            a["source"]["history_raw_end"] == OFFSET and a["source"]["prior_keys_raw"] == [0, 9] and
            a["source"]["source_row0_tokens"] == list(ROW0) and
            a["source"]["source_row1_tokens"] == list(ROW1) and
            a["source"]["A_box"] == [0, 13, 536, 999] and
            a["source"]["F_box"] == [0, 0, 33, 86] and
            a["history_write"]["coordinate_slots_raw"] == [4,5,6,7] and
            a["history_write"]["A_ids"] == list(ROW0[4:8]) and
            a["history_write"]["F_ids"] == list(ROW1[4:8]) and
            a["history_write"]["actual_changed_raw"] == [5,6,7] and
            a["history_write"]["physical_changed_slots"] == [1367,1368,1369] and
            a["history_write"]["current_tokens_supplied"] == 0 and
            a["arms"] == list(ARMS) and a["max_forwards_by_arm"] == [9,9,9,16,16] and
            a["owned_paths"][-1] == str(OUT) and
            a["max_model_forwards"] == a["max_vision_forwards"] == a["max_emitted_tokens"] == 59 and
            a["reused_calls"] == 0 and a["artifact_planning_bytes"] == 2*1024**3,
            "finite admission/endpoint changed")
    for name in ("protocol", "predecessor_acceptance", "predecessor_receipt",
                 "predecessor_cold_readback", "predecessor_verification", "predecessor_outer", "source_panel"):
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
    for entries in a["accepted_reference_vectors"].values():
        require(len(entries)==9 and all(bind(x["path"])==x for x in entries),
                "accepted reference vectors changed")
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


def step_inputs(model, batch, raw, pad, arm, emitted, *, previous=None,
                target=TARGET, prefix=OFFSET, donor=None, slots=(4,5,6,7)):
    t = len(emitted)
    expected_donor = "F" if arm in F_ARMS else "A"
    require(arm in ARMS and target == TARGET and prefix == OFFSET and
            donor in (None,expected_donor) and slots == (4,5,6,7) and
            0 <= t < (16 if arm in F_ARMS else 9) and
            (t == 0 or emitted[-1] == previous),
            "wrong target/prefix/donor/slots/own greedy token/step")
    tails = base._prefix_tokens(raw, OFFSET+t, pad)
    history = list(raw[TARGET]["token_ids"][:OFFSET])
    require(history == list(ROW0), "source history differs")
    if arm != ARMS[0]:
        history[4:8] = list((ROW1 if expected_donor == "F" else ROW0)[4:8])
    tails[TARGET] = history + list(emitted)
    histories = [list(p)+tail for p,tail in zip(batch.prompt_token_ids,tails,strict=True)]
    full = base.exact_history_inputs(model,batch.inputs,histories,pad_token_id=pad,logits_to_keep=1)
    width = int(full["input_ids"].shape[1])
    full["cache_position"] = torch.arange(width,device=full["input_ids"].device)
    require(width == PROMPT+OFFSET+t and full["attention_mask"].shape == (4,width) and
            full["position_ids"].shape == (3,4,width) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            tails[TARGET][:OFFSET] == list(ROW1 if expected_donor == "F" else ROW0) and
            tails[TARGET][OFFSET:] == list(emitted) and
            [i for i,(x,y) in enumerate(zip(tails[TARGET][:OFFSET],ROW0)) if x!=y] ==
                ([5,6,7] if expected_donor == "F" else []) and
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
    if arm in MASKED: actual[selected] = False
    verify_rectangle(native,actual,selected,arm,t)
    return (full["attention_mask"] if arm in (ARMS[0],ARMS[3]) else actual),native,selected


def verify_rectangle(native, actual, selected, arm, t):
    require(native.shape == actual.shape == selected.shape and int(selected.sum()) == 9*t and
            torch.equal(actual[~selected],native[~selected]) and
            torch.equal(actual[selected],torch.zeros_like(native[selected]) if arm in MASKED else native[selected]),
            "selected/complement mask changed")


def caller(model, full, arm, observer):
    mask,native,selected=mask_for(full,arm,source_hash=base.tensor_hash(full["attention_mask"]))
    observer.update(expected_mask=native if arm in (ARMS[0],ARMS[3]) else mask,native_mask=native,
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
            torch.equal(seen["expected_mask"],native if arm in (ARMS[0],ARMS[3]) else mask) and
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
        for arm in (ARMS if t < 9 else F_ARMS):
            full=step_inputs(model,batch,raw,pad,arm,emitted,previous=emitted[-1] if t else None)
            reference=create_causal_mask(cfg,torch.empty((*full["attention_mask"].shape,1)),
                full["attention_mask"],full["cache_position"],None,position_ids=full["position_ids"][0])
            require(torch.equal(base.native_4d(full["attention_mask"]),reference),
                    "installed SDPA causal mask changed")
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
            if t == 0:
                require(torch.equal(mask_for(full,arm)[1],seen["expected_mask"]),"t0 not native")
            if arm in F_ARMS:
                written=full["input_ids"][TARGET,PROMPT:PROMPT+OFFSET].tolist()
                source_tails=base._prefix_tokens(raw,OFFSET+t,pad)
                source_tails[TARGET]=list(ROW0)+emitted
                source_histories=[list(p)+tail for p,tail in zip(batch.prompt_token_ids,source_tails,strict=True)]
                source_full=base.exact_history_inputs(model,batch.inputs,source_histories,
                                                      pad_token_id=pad,logits_to_keep=1)
                require(written[4:8]==list(ROW1[4:8]) and
                        [i for i,(x,y) in enumerate(zip(written,ROW0)) if x!=y]==[5,6,7] and
                        torch.equal(full["position_ids"],source_full["position_ids"]) and
                        torch.equal(full["attention_mask"],source_full["attention_mask"]),
                        "joint F write/positions changed")
                checks.append(f"exact_three_F_changes_t{t}")
        arm=ARMS[4];full=step_inputs(model,batch,raw,pad,arm,emitted,
                                    previous=emitted[-1] if t else None)
        for label,kw in (("target",{"target":3}),
                         ("query",{"query":(PROMPT+OFFSET-1,PROMPT+OFFSET+t)}),
                         ("key",{"key":(PROMPT+1,PROMPT+OFFSET)})):
            try: mask_for(full,arm,**kw)
            except ValueError: checks.append(f"reject_{label}_t{t}")
            else: raise AssertionError(label)
        native=mask_for(full,arm)[1];actual=mask_for(full,arm)[0];selected=mask_for(full,arm)[2]
        bad=actual.clone();bad[0,0,-1,0]=~bad[0,0,-1,0]
        try:verify_rectangle(native,bad,selected,arm,t)
        except ValueError:checks.append(f"reject_complement_t{t}")
        else:raise AssertionError("same-rectangle complement")
        bad=full["attention_mask"].clone();bad[0,0]=1-int(bad[0,0]);badfull={**full,"attention_mask":bad}
        try:mask_for(badfull,arm,source_hash=base.tensor_hash(full["attention_mask"]))
        except ValueError:checks.append(f"reject_source_mask_t{t}")
        else:raise AssertionError("source mask")
        seen={}; class_fake=type("Fake",(),{"__call__":lambda self,**kwargs:SimpleNamespace(logits=torch.zeros((4,1,152670)))})
        _=caller(class_fake(),full,arm,seen)
        seen["actual_input"]={k:base.tensor_hash(full[k]) for k in KEYS};seen["actual_input"]["attention_mask"]=base.tensor_hash(mask_for(full,arm)[0])
        seen["layers"]=list(range(28));seen["layer_mask_hashes"]=[base.tensor_hash(seen["expected_mask"])]*28
        logits=torch.zeros((4,152670));logits[TARGET,151646]=1
        for label,k,index in (("history","input_ids",(TARGET,PROMPT)),
                              ("companion","input_ids",(0,PROMPT)),
                              ("own_prefix","input_ids",(TARGET,PROMPT+OFFSET+t-1 if t else PROMPT+OFFSET-1)),
                              ("position","position_ids",(0,TARGET,PROMPT)),
                              ("source_attention","attention_mask",(0,PROMPT))):
            mutated={**full,k:full[k].clone()};mutated[k][index]+=1
            try:verify_step(model,batch,raw,pad,arm,emitted,mutated,seen,logits,151646,
                            previous=emitted[-1] if t else None)
            except ValueError:checks.append(f"reject_{label}_t{t}")
            else:raise AssertionError(label)
        if t:
            mutated={**full,"input_ids":full["input_ids"].clone()}
            mutated["input_ids"][TARGET,PROMPT+OFFSET]+=1
            try:verify_step(model,batch,raw,pad,arm,emitted,mutated,seen,logits,151646,
                            previous=emitted[-1])
            except ValueError:checks.append(f"reject_current_header_t{t}")
            else:raise AssertionError("current header")
        try:verify_step(model,batch,raw,pad,arm,emitted,full,seen,logits,151647,
                        previous=emitted[-1] if t else None)
        except ValueError:checks.append(f"reject_wrong_greedy_t{t}")
        else:raise AssertionError("wrong greedy")
        if t:
            try:step_inputs(model,batch,raw,pad,arm,emitted,previous=-1)
            except ValueError:checks.append(f"reject_own_previous_t{t}")
            else:raise AssertionError("own previous")
        for label,kw in (("input_target",{"target":3}),
                         ("wrong_donor",{"donor":"A"}),
                         ("wrong_slot",{"slots":(4,5,6,8)})):
            try:step_inputs(model,batch,raw,pad,arm,emitted,**kw)
            except ValueError:checks.append(f"reject_{label}_t{t}")
            else:raise AssertionError(label)
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
        full=step_inputs(fixture,batch,raw,pad,ARMS[4],emitted,previous=emitted[-1] if t else None)
        widths.append(int(full["input_ids"].shape[1]))
    pixel_elements=int(batch.inputs["pixel_values"].numel())
    require(widths==list(range(1371,1387)) and pixel_elements==24502272 and
            len(checks)>=30,"finite source-shape/caller checks failed")
    cfg=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    vocab,hidden=int(cfg.vocab_size),int(cfg.hidden_size)
    measured=a["measured_predecessor"]
    require(measured["model_forwards"]==27 and measured["vision_forwards"]==27 and
            abs(measured["outer_seconds"]-79.95819828659296)<1e-9 and
            measured["raw_bytes"]==361619461,"measured predecessor changed")
    forecast=2*measured["outer_seconds"]*(59/27)*(pixel_elements/measured["pixel_elements"])*(
        (sum(widths)/len(widths))/((measured["width_min"]+measured["width_max"])/2))
    artifact_forecast=2*measured["raw_bytes"]*(59/27)+64*1024**2
    require(artifact_forecast<a["artifact_planning_bytes"],"artifact forecast exceeds envelope")
    predecessor=json.loads(base.PREFLIGHT.read_text())
    from probes.training_set_completion.recurrence_first_localization_collapse import run as prior
    prior_pre=json.loads(prior.PREFLIGHT.read_text())
    from probes.training_set_completion.recurrence_free_header_routing import run as free
    free_pre=json.loads(free.PREFLIGHT.read_text())
    direct=[Path(__file__),Path(base.__file__),Path(prior.__file__),Path(free.__file__),
            Path(inspect.getfile(create_causal_mask)),Path(inspect.getfile(modeling_qwen3_vl)),
            Path(inspect.getfile(iou_xyxy))]
    direct += [Path(c["maintained"]["path"]) for c in predecessor["captures"]]
    direct += [Path(c["maintained"]["path"]) for c in prior_pre["direct_source_captures"]]
    direct += [Path(c["maintained"]["path"]) for c in free_pre["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(direct):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_collapse_history_content.run"]
    packet={"status":"cpu_qualified_before_gpu","admission":bind(ADMISSION),"protocol":bind(PROTOCOL),
            "predecessor_acceptance":bind(a["predecessor_acceptance"]["path"]),
            "producer":bind(Path(__file__)),
            "source_bindings":a["source_bindings"],"source_identity":receipt["identity"],
            "accepted_reference_vectors":a["accepted_reference_vectors"],
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
            counts["model_forwards"]+=1;require(counts["model_forwards"]<=59,"model-forward cap")
            active["actual_input"]={k:base.tensor_hash(kwargs[k]) for k in KEYS}
            active["actual_input"]["attention_mask"]=base.tensor_hash(kwargs["attention_mask"])
            require(active["actual_input"]==active["expected_input_hashes"],"top-level actual input changed")
        def vision(_m,_args):
            counts["vision_forwards"]+=1;require(counts["vision_forwards"]<=59,"vision-forward cap")
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
                limit=16 if arm in F_ARMS else 9
                for t in range(limit):
                    previous=emitted[-1] if t else None
                    full=step_inputs(model,batch,raw,pad,arm,emitted,previous=previous)
                    mask,native,selected=mask_for(full,arm)
                    expected_mask=native if arm in (ARMS[0],ARMS[3]) else mask
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
                            role="content_native_A") for i in range(4)]
                        require(all(x["passed"] for x in entry["source_trace_parity"]) and
                                chosen==ROW1[t],"native source chosen/top2/normalizer parity failed")
                    else:
                        entry["companion_trace_parity"]=[base._trace_compare(logits=logits[i],trace=trace,
                            batch_index=i,absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],
                            role="content_companion") for i in (0,1,3)]
                        require(all(x["passed"] for x in entry["companion_trace_parity"]),
                                "active companion source parity failed")
                    if arm in (ARMS[0],ARMS[2]):
                        old=torch.load(a["accepted_reference_vectors"][arm][t]["path"],
                                       map_location="cpu",weights_only=True)
                        entry["accepted_reference_full_vector_error"]=float((logits-old["logits"]).abs().max())
                        require(entry["accepted_reference_full_vector_error"]<=TOL and
                                chosen==int(torch.argmax(old["logits"][TARGET]).item()),
                                "accepted A-history full-vector reference differs")
                    if arm!=ARMS[0]:
                        if t<9:
                            old=torch.load(OUT/f"{ARMS[0]}-step{t}.pt",map_location="cpu",weights_only=True)
                            entry["companion_state_error"]=float((payload["companions_by_layer"]-old["companions_by_layer"]).abs().max())
                            entry["companion_vector_error"]=max(float((logits[i]-old["logits"][i]).abs().max()) for i in (0,1,3))
                            require(max(entry["companion_state_error"],entry["companion_vector_error"])<=TOL,
                                    "companion states/vectors changed")
                        else:
                            entry["matched_A_reference"]="UNAVAILABLE_not_executed"
                        if arm in (ARMS[1],ARMS[2]) or (arm==ARMS[4] and t<len(receipt["arms"][3]["steps"])):
                            same=ARMS[3] if arm==ARMS[4] else ARMS[0]
                            old=torch.load(OUT/f"{same}-step{t}.pt",map_location="cpu",weights_only=True)
                            entry["same_history_state_error"]=float((payload["historical_by_layer"]-old["historical_by_layer"]).abs().max())
                            require(entry["same_history_state_error"]<=TOL,
                                    "within-written-history prior states changed")
                            if arm==ARMS[1]:
                                entry["sham_full_vector_error"]=float((logits-old["logits"]).abs().max())
                                require(entry["sham_full_vector_error"]<=TOL and chosen==ROW1[t],
                                        "independent A identity sham differs")
                            if arm==ARMS[4] and t==0:
                                entry["F_t0_full_vector_error"]=float((logits-old["logits"]).abs().max())
                                require(entry["F_t0_full_vector_error"]<=TOL and
                                        chosen==receipt["arms"][3]["steps"][0]["chosen"],
                                        "F t0 native agreement failed")
                        elif arm==ARMS[4]:
                            entry["matched_F_native_state"]="UNAVAILABLE_not_executed"
                        if arm==ARMS[2] and t==0:
                            entry["A_t0_full_vector_error"]=float((logits-old["logits"]).abs().max())
                            require(entry["A_t0_full_vector_error"]<=TOL and chosen==ROW1[0],
                                    "A t0 native agreement failed")
                    if arm==ARMS[4] and t>=1:
                        a_mask=receipt["arms"][2]
                        if t<len(a_mask["steps"]) and emitted==a_mask["emitted"][:t]:
                            a_vector=torch.load(OUT/f"{ARMS[2]}-step{t}.pt",map_location="cpu",weights_only=True)["logits"]
                            entry["blocked_equal_prefix_vector_error"]=float((logits[TARGET]-a_vector[TARGET]).abs().max())
                            require(entry["blocked_equal_prefix_vector_error"]<=TOL,
                                    "blocked equal-prefix distributions diverged")
                        else:
                            entry["blocked_equal_prefix_comparison"]="INAPPLICABLE_or_no_A_reference"
                    emitted.append(chosen);counts["emitted_target_tokens"]+=1
                    require(counts["emitted_target_tokens"]<=59,"emitted-token cap")
                    parsed=parse_row(emitted,special)
                    record["emitted"]=list(emitted);record["stop"]=parsed["stop"]
                    receipt["counts"]=dict(counts)
                    write_new(OUT/f"checkpoint-{arm}-{t}.json",receipt)
                    if parsed["stop"] is not None:break
                require(record["stop"] is not None,"trajectory did not stop by cap")
                if arm in ARMS[:3]:
                    expected_row=ROW0 if arm==ARMS[2] else ROW1
                    require(record["stop"]=="complete" and record["emitted"]==list(expected_row),
                            "A-history source/sham/mask qualification failed")
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
            r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==r["counts"]["emitted_target_tokens"]<=59 and
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
    saved={};count=0
    for record in r["arms"]:
        arm=record["arm"];emitted=[];step_summaries=[];payloads=[]
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
                    torch.equal(payload["actual_mask"],(nativemask if arm in (ARMS[0],ARMS[3]) else mask)) and
                    e["actual_mask_hash"]==base.tensor_hash(payload["actual_mask"]) and
                    e["selected_cells"]==9*t and
                    e["selected_native_hash"]==base.tensor_hash(nativemask[selected]) and
                    e["selected_actual_hash"]==base.tensor_hash(payload["actual_mask"][selected]) and
                    e["complement_native_hash"]==e["complement_actual_hash"]==base.tensor_hash(nativemask[~selected]) and
                    e["actual_layer_mask_hashes"]==[e["actual_mask_hash"]]*28,
                    "cold mask/source/consumer rectangle changed")
            seen={"actual_input":e["input_hashes"],"layers":e["actual_layers"],
                  "layer_mask_hashes":e["actual_layer_mask_hashes"],
                  "expected_mask":nativemask if arm in (ARMS[0],ARMS[3]) else mask,"selected":selected}
            verify_step(fixture,batch,raw,pad,arm,emitted,full,seen,logits,e["chosen"],
                        previous=emitted[-1] if t else None)
            if arm==ARMS[0]:
                parity=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],role="cold_content_native_A")
                    for i in range(4)]
                require(all(x["passed"] for x in parity) and e["chosen"]==ROW1[t],"cold native trace failed")
            else:
                parity=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                    absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],role="cold_content_companion")
                    for i in (0,1,3)]
                require(all(x["passed"] for x in parity),"cold companion trace failed")
                if t<9:
                    old=saved[ARMS[0]][t]
                    require(max(float((payload["companions_by_layer"]-old["companions_by_layer"]).abs().max()),
                                max(float((logits[i]-old["logits"][i]).abs().max()) for i in (0,1,3)))<=TOL,
                            "cold companion state/vector differs")
                if arm in (ARMS[1],ARMS[2]) or (arm==ARMS[4] and t<len(saved[ARMS[3]])):
                    baseline=saved[ARMS[3]][t] if arm==ARMS[4] else saved[ARMS[0]][t]
                    require(float((payload["historical_by_layer"]-
                                   baseline["historical_by_layer"]).abs().max())<=TOL,
                            "cold within-written-history states differ")
                    if arm==ARMS[1]:
                        require(float((logits-baseline["logits"]).abs().max())<=TOL and e["chosen"]==ROW1[t],
                                "cold sham differs")
                    if arm in MASKED and t==0:
                        require(float((logits-baseline["logits"]).abs().max())<=TOL,
                                "cold masked t0 differs from same-history native")
            if arm in (ARMS[0],ARMS[2]):
                reference=torch.load(a["accepted_reference_vectors"][arm][t]["path"],
                                     map_location="cpu",weights_only=True)
                require(float((logits-reference["logits"]).abs().max())<=TOL and
                        e["chosen"]==int(torch.argmax(reference["logits"][TARGET]).item()),
                        "cold accepted A reference differs")
            if (arm==ARMS[4] and t>=1 and t<len(saved[ARMS[2]]) and
                    emitted==result["arms"][2]["emitted"][:t]):
                require(float((logits[TARGET]-saved[ARMS[2]][t]["logits"][TARGET]).abs().max())<=TOL,
                        "cold equal-prefix blocked vectors diverged")
            v=logits[TARGET].double();top=torch.topk(v,2)
            step_summaries.append({"step":t,"chosen":e["chosen"],"top2_ids":top.indices.tolist(),
                "top2_logits":top.values.tolist(),"log_normalizer":float(torch.logsumexp(v,-1)),
                "chosen_logprob":float(torch.log_softmax(v,-1)[e["chosen"]]),"raw":e["raw"]})
            emitted.append(e["chosen"]);count+=1
            payloads.append(payload)
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
        saved[arm]=payloads
        if arm in ARMS[:3]:
            require(record["stop"]=="complete" and
                    emitted==list(ROW0 if arm==ARMS[2] else ROW1),
                    "cold A-source/sham/mask qualification failed")
    require(count==r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==r["counts"]["emitted_target_tokens"] and
            count<=59 and len(result["arms"])==5,"cold finite calls/arms changed")
    treatment=result["arms"][3]
    a_box=a["source"]["A_box"];f_box=a["source"]["F_box"]
    def classify(row):
        if row["stop"]!="complete":return row["stop"]
        if row["description_ids"]!=[8987]:return "other_class"
        if row["geometry"]!="valid":return "invalid_geometry"
        ia=iou_xyxy(row["box"],a_box);iff=iou_xyxy(row["box"],f_box)
        row["iou_A"]=ia;row["iou_F"]=iff
        if min(abs(ia-.5),abs(iff-.1),abs(iff-.5),abs(ia-.1))<=1e-6:return "numerical_HOLD"
        if ia>=.5 and iff<=.1:return "broad_A"
        if iff>=.5 and ia<=.1:return "fragment_F"
        return "neither_region"
    for row in result["arms"]:row["region"]=classify(row)
    outcome=treatment["region"]
    result["primary_outcome"]=outcome
    result["primary_pass"]=outcome=="broad_A"
    result["secondary_exact_row0"]=(treatment["stop"]=="complete" and treatment["emitted"]==list(ROW0))
    result["first_divergence_from_row0"]=next((i for i,(x,y) in enumerate(zip(treatment["emitted"],ROW0)) if x!=y),
                                               min(len(treatment["emitted"]),len(ROW0)))
    image=Image.open(a["source_bindings"]["image"]["path"]).convert("RGB");draw=ImageDraw.Draw(image)
    for label,box,color in (("A row0",a_box,"#00ee77"),("F row1",f_box,"#ff5533"),
                            ("F native",treatment["box"],"#00d9ff"),
                            ("F mask",result["arms"][4]["box"],"#ee00dd")):
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
