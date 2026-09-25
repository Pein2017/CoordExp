"""One fixed B-history contrast with the next row header freely generated."""

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
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.recurrence_dynamics.recurrence_history_owner_routing import run as prior


base = prior.base
require, bind, write_new = prior.require, prior.bind, prior.write_new
REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-free-header-routing"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-free-header-routing/attempt-001")
SHAS = {PROTOCOL: "1970681facd8afda15b346d17826333abc6dadb7d642aa452b0df847555e3d83",
        ADMISSION: "ea8b76470fba5f9804d02adb2249b31c88f5faf4293bc3c745084f74a369b742"}
ARMS = ("native", "identity-write-sham", "B-donor")
NATIVE = (151646,65,9605,151647,151648,151675,151867,151887,152077,151649)
BOWL_HEADER = NATIVE[:5]
TARGET, OFFSET, PROMPT = 3, 10, 1362
TOL = 2e-4


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-three-free-header-trajectories" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["raw_output_root"] == str(OUT) and
            a["original_prefix_end"] == OFFSET and a["initial_current_row_tokens_supplied"] == [] and
            a["coordinate_replacement_span"] == [5,9] and
            a["max_model_forwards"] == a["max_vision_forwards"] == a["max_emitted_target_tokens"] == 36 and
            a["max_width"] == 1387 and a["prior_sequence_gpu_hours"] == 0.29135763040168866 and
            [x["name"] for x in a["arms"]] == list(ARMS) and
            [x["max_tokens"] for x in a["arms"]] == [10,10,16] and
            tuple(a["native_expected_row"]) == NATIVE and
            a["frozen_complete_row_class_prediction"]["description_token_ids"] == [2388,41622] and
            tuple(a["conditional_bowl_header"]) == BOWL_HEADER,
            "admitted arms/bounds/prediction changed")
    for name in ("protocol", "predecessor_admission", "predecessor_acceptance", "time_update",
                 "reference_receipt", "reference_readback"):
        b = a[name]
        require(bind(b["path"])["sha256"] == b["sha256"], f"bound predecessor changed: {name}")
    for b in (list(a["source_bindings"].values()) + a["native_suffix_reference_vectors"] +
              a["conditional_B_bowl_suffix_reference_vectors"]):
        require(bind(b["path"])["sha256"] == b["sha256"], "source/reference changed")
    _, image = prior.contract()
    require(image["source_bindings"] == a["source_bindings"] and
            image["request_ids"] == a["full_batch_request_ids"] and
            image["full_batch_target_lengths"] == a["source_full_native_lengths"],
            "original full batch changed")
    return a, image


def parse_row(ids, special_ids=frozenset()):
    """Return the first-row stop and free description without restricting its words."""
    require(1 <= len(ids) <= 16, "parser length outside admitted row")
    stage = "opener"; desc = []; box = []
    for index, token in enumerate(ids):
        if token == 151645:
            return {"stop": "eos", "stage": stage, "description_ids": desc, "box_ids": box}
        if token == 151649 and stage != "terminator":
            return {"stop": "early_row_terminator", "stage": stage,
                    "description_ids": desc, "box_ids": box}
        if stage == "opener":
            if token != 151646:
                return {"stop": "malformed_opener", "stage": stage,
                        "description_ids": desc, "box_ids": box}
            stage = "description"
        elif stage == "description":
            if token == 151647:
                if not desc:
                    return {"stop": "malformed_empty_description", "stage": stage,
                            "description_ids": desc, "box_ids": box}
                stage = "box_start"
            elif 0 <= token < 151645 and token not in special_ids:
                desc.append(token)
            else:
                return {"stop": "malformed_description", "stage": stage,
                        "description_ids": desc, "box_ids": box}
        elif stage == "box_start":
            if token != 151648:
                return {"stop": "malformed_box_start", "stage": stage,
                        "description_ids": desc, "box_ids": box}
            stage = "coordinates"
        elif stage == "coordinates":
            if not 151670 <= token < 152670:
                return {"stop": "malformed_coordinate", "stage": stage,
                        "description_ids": desc, "box_ids": box}
            box.append(token)
            if len(box) == 4: stage = "terminator"
        elif stage == "terminator":
            if token != 151649:
                return {"stop": "malformed_terminator", "stage": stage,
                        "description_ids": desc, "box_ids": box}
            return {"stop": "complete", "stage": stage,
                    "description_ids": desc, "box_ids": box}
        else:
            raise AssertionError("unreachable parser stage")
    return {"stop": "cap" if len(ids) == 16 else None, "stage": stage,
            "description_ids": desc, "box_ids": box}


def step_inputs(model, batch, raw, pad, arm, emitted, *, previous=None,
                target=TARGET, slot=(5,9), donor=None):
    t = len(emitted)
    require(arm in ARMS and 0 <= t < (16 if arm == "B-donor" else 10) and
            (t == 0 or emitted[-1] == previous), "wrong arm/own previous token/step")
    history = prior.replacement(raw, arm, target=target, slot=slot, donor=donor)[:OFFSET]
    require(len(history) == OFFSET and history[:5] == [151646,65,9605,151647,151648] and
            history[9] == 151649, "historical row serialization changed")
    tails = base._prefix_tokens(raw, OFFSET+t, pad)
    tails[TARGET] = history + list(emitted)
    histories = [list(p)+tail for p,tail in zip(batch.prompt_token_ids,tails,strict=True)]
    full = base.exact_history_inputs(model,batch.inputs,histories,pad_token_id=pad,logits_to_keep=1)
    width = int(full["input_ids"].shape[1])
    full["cache_position"] = torch.arange(width,device=full["input_ids"].device)
    require(width == 1372+t and full["attention_mask"].shape == (4,width) and
            full["position_ids"].shape == (3,4,width) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            tails[2][:10] == raw[2]["token_ids"][:10] and
            tails[2][10:11] == ([raw[2]["token_ids"][10]] if t else []) and
            tails[2][11:] == [pad]*max(0,t-1) and
            tails[TARGET][OFFSET:] == list(emitted),
            "full-batch prefix/source EOS/pad/positions changed")
    return full


def verify_step(model,batch,raw,pad,arm,emitted,full,actual,logits,chosen,layers,*,previous=None):
    expected = step_inputs(model,batch,raw,pad,arm,emitted,previous=previous)
    keys=("input_ids","attention_mask","position_ids","cache_position")
    hashes={k:base.tensor_hash(expected[k]) for k in keys}
    require({k:base.tensor_hash(full[k]) for k in keys}==hashes and actual==hashes,
            "actual source/replacement/own-prefix/position/mask differs")
    require(layers==list(range(28)),"not all native SDPA masks observed")
    require(chosen==int(torch.argmax(logits[TARGET]).item()),"greedy argmax changed")
    return hashes


def cpu_checks(batch,raw,pad,special_ids):
    model=base.prior.ConfigOnlyRope(); checks=[]
    require(parse_row(NATIVE,special_ids)["stop"]=="complete" and
            parse_row(NATIVE,special_ids)["description_ids"]==[65,9605] and
            parse_row([151646,2388,41622,151647,151648,151670,151671,151672,151673,151649],special_ids)["stop"]=="complete" and
            parse_row([151645],special_ids)["stop"]=="eos" and
            parse_row([151649],special_ids)["stop"]=="early_row_terminator" and
            parse_row([151646,151647],special_ids)["stop"]=="malformed_empty_description" and
            parse_row([151646,65,151648],special_ids)["stop"]=="malformed_description" and
            parse_row([151646,65,151647,151670],special_ids)["stop"]=="malformed_box_start" and
            parse_row([151646,65,151647,151648,152670],special_ids)["stop"]=="malformed_coordinate" and
            parse_row([151646,65,151647,151648,152669],special_ids)["stop"] is None and
            parse_row([151646]+[65]*15,special_ids)["stop"]=="cap" and
            parse_row([151646,65,151647,151648,151670,151671,151672,151673,151670],special_ids)["stop"]=="malformed_terminator", 
            "canonical variable-header/EOS/early/malformed/exclusive/cap parser changed")
    checks.append("variable_header_parser_boundaries")
    for arm in ARMS:
        for t in (0,15 if arm=="B-donor" else 9):
            emitted=list(NATIVE[:t]) if t<=9 else [151646]+[65]*(t-1)
            previous=None if t==0 else emitted[-1]
            full=step_inputs(model,batch,raw,pad,arm,emitted,previous=previous)
            class Fake:
                seen=None
                def __call__(self,**kwargs):
                    self.seen={k:base.tensor_hash(kwargs[k]) for k in
                               ("input_ids","attention_mask","position_ids","cache_position")}
                    v=torch.zeros((4,152670)); v[TARGET,151649]=1
                    return SimpleNamespace(logits=v[:,None,:])
            fake=Fake(); logits=prior.forward_step(fake,full).logits[:,-1]
            verify_step(model,batch,raw,pad,arm,emitted,full,fake.seen,logits,151649,
                        list(range(28)),previous=previous)
            checks.append(f"actual_caller_receipt_{arm}_step{t}")
            for label,change in (("wrong_target",{"target":2}), ("wrong_slot",{"slot":(6,10)}),
                                 ("mutated_donor",{"donor":prior.HISTORY[2]})):
                if label=="mutated_donor" and arm=="native": continue
                try: step_inputs(model,batch,raw,pad,arm,emitted,previous=previous,**change)
                except ValueError: checks.append(f"reject_{label}_{arm}_step{t}")
                else: raise AssertionError(f"caller accepted {label}")
            for label,key,index in (("header","input_ids",(TARGET,PROMPT+OFFSET)),
                                    ("companion","input_ids",(0,PROMPT+OFFSET-1)),
                                    ("historical_slot","input_ids",(TARGET,PROMPT+5)),
                                    ("position","position_ids",(0,TARGET,PROMPT+5)),
                                    ("mask","attention_mask",(TARGET,PROMPT+5))):
                if label=="header" and t==0: continue
                bad={**full,key:full[key].clone()}; bad[key][index] += 1
                try: verify_step(model,batch,raw,pad,arm,emitted,bad,fake.seen,logits,151649,
                                 list(range(28)),previous=previous)
                except ValueError: checks.append(f"reject_{label}_{arm}_step{t}")
                else: raise AssertionError(f"receipt accepted {label}")
            if t:
                try: step_inputs(model,batch,raw,pad,arm,emitted,previous=-1)
                except ValueError: checks.append(f"reject_previous_own_token_{arm}")
                else: raise AssertionError("caller accepted changed previous token")
            try: verify_step(model,batch,raw,pad,arm,emitted,full,fake.seen,logits,151648,
                             list(range(28)),previous=previous)
            except ValueError: checks.append(f"reject_greedy_{arm}_step{t}")
            else: raise AssertionError("receipt accepted wrong greedy")
    return checks


def preflight():
    a,image=contract()
    require(not PREFLIGHT.exists() and not OUT.exists(),"attempt already prepared/run")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded language model")
    batch,raw,_trace,receipt,planning=base.source(q,image,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id); special_ids=frozenset(q.tokenizer.all_special_ids)
    checks=cpu_checks(batch,raw,pad,special_ids)
    shapes=[]
    for t in range(16):
        emitted=list(NATIVE[:t]) if t<=9 else [151646]+[65]*(t-1)
        full=step_inputs(base.prior.ConfigOnlyRope(),batch,raw,pad,"B-donor",emitted,
                         previous=None if t==0 else emitted[-1])
        shapes.append(int(full["input_ids"].shape[1]))
        if t==0:
            config=AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
            config._attn_implementation="sdpa"
            reference=create_causal_mask(config,torch.empty((*full["attention_mask"].shape,1)),
                                         full["attention_mask"],torch.arange(shapes[-1]),None,
                                         position_ids=full["position_ids"][0])
            require(torch.equal(base.prior.native_4d(full["attention_mask"]),reference),
                    "native SDPA mask differs from installed consumer")
    require(shapes==list(range(1372,1388)) and
            int(batch.inputs["pixel_values"].numel())==24403968,
            "full source/maximum vision shape changed")
    forecast=2*57.915783612*(36/20)
    require(abs(forecast-a["planning_outer_gpu_seconds"])<1e-6,
            "shape-aware planning forecast changed")
    raw_forecast=2*123113506*(36/20)
    require(raw_forecast<a["artifact_plan_bytes"],"artifact planning envelope exceeded")
    predecessor=json.loads(prior.PREFLIGHT.read_text())
    direct=[Path(__file__),Path(prior.__file__),Path(base.__file__),
            Path(inspect.getfile(create_causal_mask)),Path(inspect.getfile(modeling_qwen3_vl))]
    direct += [Path(c["maintained"]["path"]) for c in predecessor["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(direct):
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved=base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.recurrence_dynamics.recurrence_free_header_routing.run"]
    packet={"schema":"free_header_routing.preflight.v1","status":"cpu_qualified_before_gpu",
            "admission":bind(ADMISSION),"protocol":bind(PROTOCOL),"producer":bind(Path(__file__)),
            "source_bindings":a["source_bindings"],"source_identity":receipt["identity"],
            "input_identity":receipt["input_identity"],"planning":planning,
            "widths":shapes,"pixel_elements":int(batch.inputs["pixel_values"].numel()),
            "pad_id":pad,"special_ids":sorted(special_ids),"cpu_checks":checks,
            "forecast_outer_seconds":forecast,"artifact_forecast_bytes":raw_forecast,
            "direct_source_captures":captures,
            "commands":{"gpu":command+["run"],"readback":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),"captures":len(captures),
                      "widths":[shapes[0],shapes[-1]],"forecast_seconds":forecast,
                      "artifact_bytes":raw_forecast}))


def checked_preflight():
    a,image=contract(); p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["admission"]==bind(ADMISSION) and
            p["protocol"]==bind(PROTOCOL) and p["producer"]==bind(Path(__file__)),
            "preflight/producer changed")
    for c in p["direct_source_captures"]:
        require(bind(c["maintained"]["path"])==c["maintained"] and
                bind(c["capture"]["path"])==c["capture"],"direct source capture changed")
    return a,image,p


def run():
    a,image,p=checked_preflight()
    require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
            "attempt already launched; no retry")
    started=time.monotonic(); device=torch.device("cuda:0"); handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"emitted_target_tokens":0}
    receipt={"schema":"free_header_routing.attempt.v1","status":"running",
             "pid":os.getpid(),"begun_unix":time.time(),"admission":bind(ADMISSION),
             "preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),"counts":counts,"arms":[]}
    write_new(OUT/"launch.json",receipt)
    try:
        torch.cuda.set_device(device); torch.empty(1,device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q,identity=base.load_model("untied",device)
        expected=p["source_identity"]
        require({k:v for k,v in identity.items() if k!="loader_source"}==
                {k:v for k,v in expected.items() if k!="loader_source"} and
                all(identity["loader_source"][k]==expected["loader_source"][k]
                    for k in ("sha256","size_bytes")),"effective model identity changed")
        model=q.model.eval()
        batch,raw,trace,source_receipt,planning=base.source(q,image,device)
        require(source_receipt["input_identity"]==p["input_identity"] and
                planning==p["planning"],"GPU source differs from CPU preparation")
        pad=int(q.tokenizer.pad_token_id); special_ids=frozenset(q.tokenizer.all_special_ids)
        require(pad==p["pad_id"] and sorted(special_ids)==p["special_ids"],
                "tokenizer pad/special IDs changed")
        layers=[m for m in model.modules() if isinstance(m,modeling_qwen3_vl.Qwen3VLTextDecoderLayer)]
        attentions=[m for m in model.modules() if isinstance(m,modeling_qwen3_vl.Qwen3VLTextAttention)]
        require(len(layers)==len(attentions)==28 and
                [m.self_attn for m in layers]==attentions,"actual 28-layer route changed")
        active={"full":None,"expected_mask":None,"seen":[],"history":[],"companions":[],
                "actual_input":None}
        def top_hook(_module,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=36,"model-forward cap")
            full=active["full"]; keys=("input_ids","attention_mask","position_ids","cache_position")
            require(all(torch.equal(kwargs[k],full[k]) for k in keys),
                    "actual forward input differs from checked caller")
            active["actual_input"]={k:base.tensor_hash(kwargs[k]) for k in keys}
        def vision_hook(_module,_args):
            counts["vision_forwards"]+=1
            require(counts["vision_forwards"]<=36,"vision-forward cap")
        handles += [model.register_forward_pre_hook(top_hook,with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(vision_hook)]
        for i,attn in enumerate(attentions):
            def attn_hook(_module,_args,kwargs,layer=i):
                mask=kwargs.get("attention_mask")
                require(isinstance(mask,torch.Tensor) and mask.ndim==4 and
                        torch.equal(mask,active["expected_mask"]),
                        f"layer {layer} consumed nonnative mask")
                active["seen"].append(layer)
            handles.append(attn.register_forward_pre_hook(attn_hook,with_kwargs=True))
        for i,layer in enumerate(layers):
            def layer_hook(_module,_args,output,layer_idx=i):
                states=output[0] if isinstance(output,tuple) else output
                require(isinstance(states,torch.Tensor) and states.ndim==3,
                        "historical/companion states unavailable")
                active["history"].append(states[TARGET,PROMPT:PROMPT+10].detach().cpu().float())
                active["companions"].append(states[[0,1,2],-1].detach().cpu().float())
            handles.append(layer.register_forward_hook(layer_hook))
        receipt["effective_identity"]=identity
        with torch.inference_mode():
            for arm in ARMS:
                emitted=[]; record={"arm":arm,"steps":[],"stop":None}
                receipt["arms"].append(record)
                limit=16 if arm=="B-donor" else 10
                for t in range(limit):
                    previous=None if t==0 else record["steps"][-1]["chosen"]
                    full=step_inputs(model,batch,raw,pad,arm,emitted,previous=previous)
                    active.update(full=full,expected_mask=base.prior.native_4d(full["attention_mask"]),
                                  seen=[],history=[],companions=[],actual_input=None)
                    out=prior.forward_step(model,full)
                    torch.cuda.synchronize(device)
                    require(len(active["history"])==len(active["companions"])==28 and
                            active["actual_input"] is not None,"consumer/state capture incomplete")
                    logits=out.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape==(4,152670) and torch.isfinite(logits).all().item(),
                            "invalid full-vocabulary vectors")
                    chosen=int(torch.argmax(logits[TARGET]).item())
                    hashes=verify_step(model,batch,raw,pad,arm,emitted,full,active["actual_input"],
                                       logits,chosen,active["seen"],previous=previous)
                    payload={"arm":arm,"step":t,"logits":logits,
                             "prior_target_by_layer":torch.stack(active["history"]),
                             "companion_last_by_layer":torch.stack(active["companions"])}
                    raw_path=OUT/f"{arm}-step{t}.pt"; torch.save(payload,raw_path)
                    input_path=OUT/f"inputs-{arm}-step{t}.json"
                    write_new(input_path,{k:full[k].detach().cpu().tolist() for k in hashes})
                    entry={"step":t,"raw":bind(raw_path),"inputs":bind(input_path),
                           "input_hashes":hashes,"actual_mask_layers":active["seen"],
                           "actual_native_mask_sha256":base.tensor_hash(active["expected_mask"]),
                           "chosen":chosen,"elapsed_internal_seconds":time.monotonic()-started}
                    if arm=="native":
                        active_rows=[i for i,r in enumerate(raw) if len(r["token_ids"])>OFFSET+t]
                        entry["source_trace_parity"]=[base._trace_compare(
                            logits=logits[i],trace=trace,batch_index=i,absolute_offset=OFFSET+t,
                            token_id=int(raw[i]["token_ids"][OFFSET+t]),role="free_header_routing")
                            for i in active_rows]
                        require(active_rows==([0,1,2,3] if t==0 else [0,1,3]) and
                                all(x["passed"] for x in entry["source_trace_parity"]) and
                                chosen==NATIVE[t],"native source/greedy parity failed")
                        if t>=5:
                            ref=a["native_suffix_reference_vectors"][t-5]
                            old=torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]
                            entry["accepted_native_suffix_max_error"]=float((logits-old).abs().max())
                            require(entry["accepted_native_suffix_max_error"]<=TOL,
                                    "accepted native suffix full-vector mismatch")
                    else:
                        active_companions=[i for i in (0,1,2) if len(raw[i]["token_ids"])>OFFSET+t]
                        entry["companion_source_trace_parity"]=[base._trace_compare(
                            logits=logits[i],trace=trace,batch_index=i,absolute_offset=OFFSET+t,
                            token_id=int(raw[i]["token_ids"][OFFSET+t]),role="free_header_companion")
                            for i in active_companions]
                        require(all(x["passed"] for x in entry["companion_source_trace_parity"]),
                                "active companion source trace changed")
                        if t<10:
                            native=torch.load(OUT/f"native-step{t}.pt",map_location="cpu",weights_only=True)
                            entry["companion_hidden_max_error"]=float((payload["companion_last_by_layer"]-
                                                                         native["companion_last_by_layer"]).abs().max())
                            entry["companion_vector_max_errors"]=[float((logits[i]-native["logits"][i]).abs().max())
                                                                    for i in (0,1,2)]
                            require(entry["companion_hidden_max_error"]<=TOL and
                                    max(entry["companion_vector_max_errors"])<=TOL,
                                    "companion states/vectors changed")
                            if arm=="identity-write-sham":
                                entry["all_batch_vector_max_error"]=float((logits-native["logits"]).abs().max())
                                entry["prior_history_max_error"]=float((payload["prior_target_by_layer"]-
                                                                           native["prior_target_by_layer"]).abs().max())
                                require(max(entry["all_batch_vector_max_error"],entry["prior_history_max_error"])<=TOL
                                        and chosen==NATIVE[t],"identity-write sham differs from native")
                        if arm=="B-donor" and 5<=t<10 and emitted[:5]==list(BOWL_HEADER):
                            ref=a["conditional_B_bowl_suffix_reference_vectors"][t-5]
                            old=torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]
                            entry["conditional_B_bowl_vector_max_error"]=float((logits-old).abs().max())
                            require(entry["conditional_B_bowl_vector_max_error"]<=TOL and
                                    chosen==a["conditional_B_bowl_tail"][t-5],
                                    "conditional freely selected bowl suffix differs")
                    record["steps"].append(entry); emitted.append(chosen)
                    counts["emitted_target_tokens"]+=1
                    require(counts["emitted_target_tokens"]<=36,"emitted-token cap")
                    parsed=parse_row(emitted,special_ids)
                    record["emitted"]=list(emitted); record["stop"]=parsed["stop"]
                    record["last_parse"]=parsed
                    receipt["counts"]=dict(counts)
                    write_new(OUT/f"checkpoint-{arm}-{t}.json",receipt)
                    if record["stop"] is not None: break
                require(record["stop"] is not None,"trajectory failed to stop by arm cap")
                if arm in ARMS[:2]:
                    require(record["emitted"]==list(NATIVE) and record["stop"]=="complete",
                            "native/sham free-header qualification failed")
        receipt["status"]="candidate_complete"
    except BaseException as exc:
        receipt["status"]="technical_invalid"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    finally:
        for h in handles: h.remove()
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        receipt["allocated_internal_seconds"]=time.monotonic()-started
        receipt["counts"]=dict(counts)
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["terminal_pid"]=os.getpid()
        receipt["artifact_bytes"]=sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        write_new(OUT/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"counts":counts,
                      "internal_seconds":receipt["allocated_internal_seconds"],
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_complete","technical failure; no retry")


def readback():
    a,image,p=checked_preflight()
    r=json.loads((OUT/"receipt.json").read_text())
    require(r["status"]=="candidate_complete" and r["admission"]==bind(ADMISSION) and
            [x["arm"] for x in r["arms"]]==list(ARMS) and
            r["counts"]["model_forwards"]==r["counts"]["vision_forwards"]==
            r["counts"]["emitted_target_tokens"]<=36 and
            not (OUT/"readback.json").exists(),"receipt/count/readback changed")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"readback loaded language model")
    batch,raw,trace,source_receipt,planning=base.source(q,image,torch.device("cpu"))
    require(source_receipt["input_identity"]==p["input_identity"] and planning==p["planning"],
            "cold original source changed")
    pad=int(q.tokenizer.pad_token_id); special_ids=frozenset(q.tokenizer.all_special_ids)
    require(pad==p["pad_id"] and sorted(special_ids)==p["special_ids"],
            "cold tokenizer ID set changed")
    model=base.prior.ConfigOnlyRope(); native_payloads=[]
    summary={"schema":"free_header_routing.readback.v1","status":"candidate_cold_readback_passed",
             "receipt":bind(OUT/"receipt.json"),"admission":bind(ADMISSION),"arms":[]}
    for record in r["arms"]:
        arm=record["arm"]; emitted=[]; rows=[]
        for t,entry in enumerate(record["steps"]):
            require(entry["step"]==t and entry["actual_mask_layers"]==list(range(28)) and
                    bind(entry["raw"]["path"])==entry["raw"] and
                    bind(entry["inputs"]["path"])==entry["inputs"],
                    "raw/actual-consumer binding changed")
            values=json.loads(Path(entry["inputs"]["path"]).read_text())
            full={k:torch.tensor(v,dtype=torch.long) for k,v in values.items()}
            payload=torch.load(entry["raw"]["path"],map_location="cpu",weights_only=True)
            require(payload["arm"]==arm and payload["step"]==t and
                    payload["logits"].shape==(4,152670) and torch.isfinite(payload["logits"]).all().item() and
                    payload["prior_target_by_layer"].shape[0]==28 and
                    payload["companion_last_by_layer"].shape[:2]==(28,3),
                    "cold raw vectors/states incomplete")
            logits=payload["logits"]
            hashes=verify_step(model,batch,raw,pad,arm,emitted,full,entry["input_hashes"],
                               logits,entry["chosen"],entry["actual_mask_layers"],
                               previous=None if t==0 else emitted[-1])
            require(hashes==entry["input_hashes"] and
                    entry["actual_native_mask_sha256"]==base.tensor_hash(base.prior.native_4d(full["attention_mask"])),
                    "cold native mask/source/own input changed")
            if arm=="native":
                active_rows=[i for i,x in enumerate(raw) if len(x["token_ids"])>OFFSET+t]
                parity=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                            absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],
                            role="cold_free_header_routing") for i in active_rows]
                require(all(x["passed"] for x in parity) and entry["chosen"]==NATIVE[t],
                        "cold native source trace/greedy failed")
                if t>=5:
                    ref=a["native_suffix_reference_vectors"][t-5]
                    old=torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]
                    require(float((logits-old).abs().max())<=TOL,
                            "cold accepted native suffix vector mismatch")
                native_payloads.append(payload)
            else:
                active_companions=[i for i in (0,1,2) if len(raw[i]["token_ids"])>OFFSET+t]
                parity=[base._trace_compare(logits=logits[i],trace=trace,batch_index=i,
                            absolute_offset=OFFSET+t,token_id=raw[i]["token_ids"][OFFSET+t],
                            role="cold_free_header_companion") for i in active_companions]
                require(all(x["passed"] for x in parity),"cold active companion source trace failed")
                if t<10:
                    native=native_payloads[t]
                    require(float((payload["companion_last_by_layer"]-
                                   native["companion_last_by_layer"]).abs().max())<=TOL and
                            all(float((logits[i]-native["logits"][i]).abs().max())<=TOL for i in (0,1,2)),
                            "cold companion vector/layer state mismatch")
                    if arm=="identity-write-sham":
                        require(float((logits-native["logits"]).abs().max())<=TOL and
                                float((payload["prior_target_by_layer"]-
                                       native["prior_target_by_layer"]).abs().max())<=TOL,
                                "cold identity-write sham mismatch")
                if arm=="B-donor" and 5<=t<10 and emitted[:5]==list(BOWL_HEADER):
                    ref=a["conditional_B_bowl_suffix_reference_vectors"][t-5]
                    old=torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]
                    require(float((logits-old).abs().max())<=TOL and
                            entry["chosen"]==a["conditional_B_bowl_tail"][t-5],
                            "cold conditional bowl suffix mismatch")
            v=logits[TARGET].double(); top=torch.topk(v,2); probs=torch.softmax(v,-1)
            rows.append({"step":t,"chosen":entry["chosen"],"top2_ids":top.indices.tolist(),
                         "top2_logits":top.values.tolist(),"gap":float(top.values[0]-top.values[1]),
                         "chosen_prob":float(probs[entry["chosen"]]),
                         "chosen_logprob":float(torch.log_softmax(v,-1)[entry["chosen"]]),
                         "raw":entry["raw"]})
            emitted.append(entry["chosen"])
            parsed=parse_row(emitted,special_ids)
            require(parsed["stop"] is None if t<len(record["steps"])-1 else
                    parsed["stop"]==record["stop"],"cold row parser/stop changed")
        require(emitted==record["emitted"],"cold own generated history changed")
        parsed=parse_row(emitted,special_ids)
        desc=q.tokenizer.decode(parsed["description_ids"],skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        box=[x-151670 for x in parsed["box_ids"]] if len(parsed["box_ids"])==4 else None
        geometry=("valid" if box[0]<box[2] and box[1]<box[3] else "invalid") if box else "no_complete_box"
        summary["arms"].append({"arm":arm,"emitted":emitted,"stop":record["stop"],
                                "description_ids":parsed["description_ids"],"description_text":desc,
                                "box":box,"geometry":geometry,"steps":rows})
    donor=summary["arms"][2]
    summary["strict_scissors_complete_prediction"]=(donor["stop"]=="complete" and
        donor["description_ids"]==a["frozen_complete_row_class_prediction"]["description_token_ids"] and
        donor["geometry"]=="valid")
    summary["freely_selected_bowl_header"]=(donor["emitted"][:5]==list(BOWL_HEADER))
    summary["conditional_bowl_reference_applicable"]=summary["freely_selected_bowl_header"]
    summary["counts"]=r["counts"]
    require(r["artifact_bytes"]<a["artifact_plan_bytes"],"artifact envelope exceeded")
    write_new(OUT/"readback.json",summary)
    print(json.dumps({"status":summary["status"],
                      "rows":{x["arm"]:x["emitted"] for x in summary["arms"]},
                      "descriptions":{x["arm"]:x["description_text"] for x in summary["arms"]},
                      "stops":{x["arm"]:x["stop"] for x in summary["arms"]},
                      "scissors_prediction":summary["strict_scissors_complete_prediction"],
                      "bowl_reference_applicable":summary["conditional_bowl_reference_applicable"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("preflight","run","readback"))
    {"preflight":preflight,"run":run,"readback":readback}[parser.parse_args().action]()
