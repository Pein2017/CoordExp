"""Six finite full-prefix y1 policy arms on one original A history."""
from __future__ import annotations

import argparse
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

from probes.training_set_completion.recurrence_history_read_stage import run as prior
from probes.training_set_completion.recurrence_free_header_routing.run import parse_row
from src.data.geometry import iou_xyxy


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-y1-row-routing"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-y1-row-routing/attempt-001")
SHAS = {PROTOCOL: "393b5246a05807f497fe9637211f710bd8c3ab1c658fa2c5eb93455228b9224a",
        ADMISSION: "a05f595e8d16b61a89354bf26c1d28ee063e540bcc6ff616355555838d8ec5ea"}
ARMS = ("native_A", "coordinate_A", "native_sham", "coordinate_sham",
        "native_flip", "coordinate_flip")
TARGET, OFFSET, PROMPT, Y1, TOL = 2, 9, 1362, 1376, 2e-4
KEYS = prior.KEYS
base = prior.base
require, bind, write_new = base.require, base.bind, base.write_new


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-finite-reciprocal-y1-row-routing" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            [x["name"] for x in a["arms"]] == list(ARMS) and
            [x["max_tokens"] for x in a["arms"]] == [9]*4+[16]*2 and
            [x["override_token"] for x in a["arms"]] ==
                [None, None, 151670, 151683, 151683, 151670] and
            [x["mask_mode"] for x in a["arms"]] ==
                ["native", "coordinate"]*3 and
            a["source"]["target_index"] == TARGET and
            a["source"]["history_raw_end"] == OFFSET and
            a["source"]["A_tokens"] == list(prior.old.ROW0) and
            a["selection"]["current_step"] == 5 and
            a["selection"]["physical_index"] == Y1 and
            a["counts"]["maximum_model_forwards"] ==
                a["counts"]["maximum_vision_forwards"] ==
                a["counts"]["maximum_logical_emissions"] == 68 and
            a["counts"]["explicit_policy_writes"] == 4 and
            a["planning"]["artifact_planning_envelope_bytes"] == 2*1024**3 and
            a["owned_paths"]["raw"] == str(OUT), "admitted six-arm scope changed")
    for name in ("protocol", "cpu_acceptance", "feasibility_report", "reference_bindings",
                 "predecessor_acceptance", "predecessor_receipt", "predecessor_readback",
                 "predecessor_outer", "predecessor_verification", "predecessor_producer",
                 "source_panel"):
        require(bind(a[name]["path"]) == a[name], f"bound predecessor changed: {name}")
    for name in ("raw", "trace", "runtime_receipt", "image"):
        require(bind(a["source_bindings"][name]["path"]) == a["source_bindings"][name],
                f"source changed: {name}")
    for name, item in a["maintained_routes"].items():
        if isinstance(item, dict):
            require(bind(item["path"]) == item, f"maintained route changed: {name}")
    cross = a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"]) == cross["maintained"] and
            cross["maintained"]["sha256"] == cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"]) == cross["prior_acceptance"],
            "maintained loader crosswalk changed")
    old_a, _, _ = prior.contract()
    require(a["source_bindings"]["raw"] == old_a["source_bindings"]["raw"] and
            a["source_bindings"]["trace"] == old_a["source_bindings"]["trace"] and
            a["source_bindings"]["source_identity"] == old_a["source_bindings"]["source_identity"] and
            all(a["region_rule"][k] == old_a["region_rule"][k] for k in
                ("A","F","class_token_ids","class_name","strict_geometry",
                 "same_region_iou_min","other_region_iou_max","numerical_guard")),
            "source or numerical regions changed")
    for arm in ("native_A", "coordinate_A"):
        require(len(a["references"][arm]) == 9, "accepted nine-step arm missing")
        for t, item in enumerate(a["references"][arm]):
            require(item["step"] == t and
                    bind(item["raw"]["path"]) == item["raw"] and
                    bind(item["inputs"]["path"]) == item["inputs"],
                    "accepted vector/input binding changed")
    return a, old_a


def spec(a, arm):
    require(arm in ARMS, "unknown arm")
    return a["arms"][ARMS.index(arm)]


def alias(config):
    return "native_A" if config["mask_mode"] == "native" else "coordinate_A"


def step_inputs(model, batch, raw, pad, config, emitted, *, previous=None,
                target=TARGET, prefix=OFFSET):
    t = len(emitted)
    require(config["name"] in ARMS and config["history"] == "A" and
            target == TARGET and prefix == OFFSET and 0 <= t < config["max_tokens"] and
            (t == 0 or emitted[-1] == previous),
            "target/history/step/own previous token changed")
    tails = base._prefix_tokens(raw, OFFSET+t, pad)
    tails[TARGET] = list(prior.old.ROW0) + list(emitted)
    histories = [list(prompt)+tail for prompt, tail in
                 zip(batch.prompt_token_ids, tails, strict=True)]
    full = base.exact_history_inputs(model, batch.inputs, histories,
                                     pad_token_id=pad, logits_to_keep=1)
    width = PROMPT+OFFSET+t
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    require(full["input_ids"].shape == full["attention_mask"].shape == (4,width) and
            full["position_ids"].shape == (3,4,width) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            tails[TARGET][:OFFSET] == list(prior.old.ROW0) and
            all(tails[i] == raw[i]["token_ids"][:OFFSET+t] for i in (0,1,3)) and
            full["attention_mask"][:,PROMPT:].all().item() and
            all(int(full["attention_mask"][i,:PROMPT].sum()) ==
                len(batch.prompt_token_ids[i]) for i in range(4)),
            "original A-history/companion/source/position changed")
    return full


def select_token(a, config, t, emitted, logits, *, raw_argmax=None, selected=None):
    require(config["name"] in ARMS and config == spec(a,config["name"]) and
            0 <= t < config["max_tokens"] and
            logits.shape == (4,152670) and torch.isfinite(logits).all().item(),
            "policy arm/step/full vector changed")
    greedy = int(torch.argmax(logits[TARGET]).item())
    if t <= 5:
        expected = a["references"][config["reference_arm"]][t]
        require(list(emitted) == [x["chosen"] for x in
                a["references"][config["reference_arm"]][:t]] and
                greedy == expected["chosen"],
                "matched free header/x1/prewrite y1 differs")
    wanted = config["override_token"] if t == 5 and config["override_token"] is not None else greedy
    policy = ("identity_write" if t == 5 and config["override_token"] == greedy else
              "flip_write" if t == 5 and config["override_token"] is not None else "greedy")
    require((raw_argmax is None or raw_argmax == greedy) and
            (selected is None or selected == wanted) and
            (t != 5 or list(emitted) == a["selection"]["before_write_prefix"]) and
            (t != 5 or greedy ==
             a["selection"]["native_raw_argmax" if config["mask_mode"] == "native"
                                                   else "coordinate_raw_argmax"]),
            "raw argmax/selected y1/one policy write changed")
    return greedy, wanted, policy


def verify_online(model, batch, raw, pad, a, old_a, config, emitted, full,
                  seen, logits, greedy, selected, policy):
    t = len(emitted)
    expected = step_inputs(model,batch,raw,pad,config,emitted,
                           previous=emitted[-1] if t else None)
    mask,native,blocked = prior.mask_for(expected,old_a,alias(config))
    actual = native if config["mask_mode"] == "native" else mask
    raw_id,chosen,expected_policy = select_token(a,config,t,emitted,logits,
                                                  raw_argmax=greedy,selected=selected)
    require(policy == expected_policy and raw_id == greedy and chosen == selected and
            all(torch.equal(full[k],expected[k]) for k in KEYS) and
            seen["actual_input"] == prior.input_hashes(expected,mask) and
            seen["layers"] == list(range(28)) and
            seen["layer_mask_hashes"] == [base.tensor_hash(actual)]*28 and
            torch.equal(seen["expected_mask"],actual) and
            torch.equal(seen["selected"],blocked) and
            (t == 0 or int(full["input_ids"][TARGET,PROMPT+OFFSET+t-1]) == emitted[-1]) and
            (t < 6 or (int(full["input_ids"][TARGET,Y1]) == emitted[5] and
                         [int(x) for x in full["position_ids"][:,TARGET,Y1]] == [401]*3)),
            "actual caller/mask/source/policy/own token consumption changed")
    return mask,native,blocked


def entry_for(full,mask,native,blocked,seen,greedy,selected,policy,t,rawpath,inputpath,previous):
    entry = prior.entry_for(full,mask,native,blocked,seen,selected,t,rawpath,inputpath)
    entry.update(raw_argmax=greedy,selected=selected,policy=policy,consumed_previous=previous)
    return entry


def verify_serialized(model,batch,raw,pad,a,old_a,config,emitted,entry,stored,payload):
    t = len(emitted)
    require(entry["step"] == t and payload["arm"] == config["name"] and
            payload["step"] == t and entry["consumed_previous"] ==
                (emitted[-1] if t else None), "serialized arm/step/previous token changed")
    expected = step_inputs(model,batch,raw,pad,config,emitted,
                           previous=emitted[-1] if t else None)
    mask,native,blocked = prior.mask_for(expected,old_a,alias(config))
    actual = native if config["mask_mode"] == "native" else mask
    logits = payload["logits"]
    greedy,selected,policy = select_token(a,config,t,emitted,logits,
                                          raw_argmax=entry["raw_argmax"],
                                          selected=entry["selected"])
    require(all(torch.equal(stored[k].detach().cpu(),expected[k].detach().cpu()) for k in KEYS) and
            entry["input_hashes"] == prior.input_hashes(expected,mask) and
            entry["actual_layers"] == list(range(28)) and
            entry["actual_layer_mask_hashes"] == [base.tensor_hash(actual)]*28 and
            torch.equal(payload["actual_mask"].detach().cpu(),actual.detach().cpu()) and
            entry["actual_mask_hash"] == base.tensor_hash(actual) and
            entry["selected_cells"] == int(blocked.sum()) and
            entry["selected_native_hash"] == base.tensor_hash(native[blocked]) and
            entry["selected_actual_hash"] == base.tensor_hash(actual[blocked]) and
            entry["complement_native_hash"] == entry["complement_actual_hash"] ==
                base.tensor_hash(native[~blocked]) and
            logits.shape == (4,152670) and torch.isfinite(logits).all().item() and
            entry["raw_argmax"] == greedy and entry["selected"] == selected and
            entry["chosen"] == selected and entry["policy"] == policy and
            (t < 6 or (int(expected["input_ids"][TARGET,Y1]) == emitted[5] and
                         [int(x) for x in expected["position_ids"][:,TARGET,Y1]] == [401]*3)),
            "serialized source/mask/raw-versus-selected/own input differs")
    return expected


def reference(a,config,t,full,logits,greedy):
    if t >= (6 if config["name"].endswith("flip") else 9):
        return None
    item = a["references"][config["reference_arm"]][t]
    require(all(json.loads(Path(item["inputs"]["path"]).read_text())[k] ==
                full[k].detach().cpu().tolist() for k in KEYS),
            "accepted same-mode full input differs")
    saved = torch.load(item["raw"]["path"],map_location="cpu",weights_only=True)["logits"]
    error = float((logits-saved).abs().max())
    require(error <= TOL and greedy == item["chosen"],
            "accepted same-mode full vector/raw argmax differs")
    return error


def compare_states(config,t,payload,logits,baselines):
    first = baselines.get("native_A")
    if first is None:
        return
    require(float((payload["historical_by_layer"]-first[0]["historical_by_layer"]).abs().max()) <= TOL,
            "prior A-history states changed")
    if config["name"] in ("native_sham","coordinate_sham"):
        base_row = baselines[config["reference_arm"]][t]
        require(torch.equal(logits,base_row["logits"]) and
                torch.equal(payload["historical_by_layer"],base_row["historical_by_layer"]) and
                torch.equal(payload["companions_by_layer"],base_row["companions_by_layer"]),
                "identity-y1 sham differs from fresh same-mode base")
    if config["name"] != "native_A" and t < 9:
        matched = first[t]
        require(max(float((payload["companions_by_layer"]-
                           matched["companions_by_layer"]).abs().max()),
                    max(float((logits[i]-matched["logits"][i]).abs().max()) for i in (0,1,3))) <= TOL,
                "matched-step companion states/vectors changed")


def trace_check(logits,trace,raw,config,t):
    target_source = (config["mask_mode"] == "native" and
                     (config["name"] != "native_flip" or t <= 5))
    return prior.trace_check(logits,trace,raw,
                             "native_A" if target_source else "coordinate_A",t)


def receipt_guard(records,a):
    require(isinstance(records,list) and [x["arm"] for x in records] == list(ARMS),
            "missing/reordered arm receipt")
    for row,config in zip(records,a["arms"],strict=True):
        require(row["mask_mode"] == config["mask_mode"] and
                row["override_token"] == config["override_token"] and
                isinstance(row["steps"],list) and
                [x["step"] for x in row["steps"]] == list(range(len(row["steps"]))) and
                1 <= len(row["steps"]) <= config["max_tokens"],
                "serialized mode/policy/steps changed")


def cpu_checks(a,old_a,batch,raw,pad,special):
    fixture = base.ConfigOnlyRope()
    cfg = AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    checks = []
    require(parse_row(list(prior.old.ROW0),special)["stop"] == "complete" and
            parse_row(list(prior.old.ROW1),special)["stop"] == "complete" and
            parse_row([151645],special)["stop"] == "eos" and
            parse_row([151649],special)["stop"] == "early_row_terminator" and
            parse_row([151646,8987,151647,151648,152670],special)["stop"] ==
                "malformed_coordinate" and
            parse_row([151646]+[8987]*15,special)["stop"] == "cap",
            "parser/upper-exclusive coordinate boundary changed")
    checks.append("parser_complete_eos_early_exclusive_cap")

    def emitted_for(config,t):
        row = [x["chosen"] for x in a["references"][config["reference_arm"]]]
        if t <= 9:
            output = row[:t]
        else:
            output = row[:8]+[151670]*(t-8)
        if t > 5 and config["override_token"] is not None:
            output[5] = config["override_token"]
        return output

    examples = [("native_A",0),("coordinate_A",1),("native_sham",5),
                ("coordinate_sham",6),("native_flip",5),("native_flip",6),
                ("coordinate_flip",5),("coordinate_flip",6),
                ("native_flip",15),("coordinate_flip",15)]
    for name,t in examples:
        config = spec(a,name);emitted = emitted_for(config,t)
        full = step_inputs(fixture,batch,raw,pad,config,emitted,
                           previous=emitted[-1] if t else None)
        mode = alias(config)
        mask,native,blocked = prior.mask_for(full,old_a,mode)
        installed = create_causal_mask(cfg,torch.empty((*full["attention_mask"].shape,1)),
            full["attention_mask"],full["cache_position"],None,
            position_ids=full["position_ids"][0])
        require(torch.equal(native,installed),"installed SDPA native mask changed")
        chosen = (a["references"][config["reference_arm"]][t]["chosen"]
                  if t <= 5 else 151670)
        logits = torch.zeros((4,152670));logits[TARGET,chosen] = 1
        seen = {}
        class Fake:
            def __call__(self,**kwargs):
                seen["actual_input"] = prior.input_hashes(kwargs,kwargs["attention_mask"])
                return SimpleNamespace(logits=logits[:,None,:])
        prior.caller(Fake(),full,old_a,mode,seen)
        seen["layers"] = list(range(28))
        seen["layer_mask_hashes"] = [base.tensor_hash(seen["expected_mask"])]*28
        greedy,selected,policy = select_token(a,config,t,emitted,logits)
        verify_online(fixture,batch,raw,pad,a,old_a,config,emitted,full,seen,
                      logits,greedy,selected,policy)
        payload = {"arm":name,"step":t,"actual_mask":seen["expected_mask"],
                   "logits":logits}
        entry = entry_for(full,mask,native,blocked,seen,greedy,selected,policy,
                          t,Path(__file__),Path(__file__),emitted[-1] if t else None)
        stored = {k:full[k].clone() for k in KEYS}
        verify_serialized(fixture,batch,raw,pad,a,old_a,config,emitted,entry,stored,payload)
        checks.append(f"actual_caller_reader_{name}_t{t}")

        if t == 5 and name in ("native_flip","coordinate_flip"):
            for label,bad in (("missing_y1_write",{**entry,"selected":greedy,"chosen":greedy}),
                              ("missing_y1_policy",{**entry,"policy":"greedy"})):
                try:verify_serialized(fixture,batch,raw,pad,a,old_a,config,emitted,
                                      bad,stored,payload)
                except ValueError:checks.append("reject_"+label)
                else:raise AssertionError(label)
            wrong = {**config,"override_token":greedy}
            try:select_token(a,wrong,t,emitted,logits)
            except ValueError:checks.append("reject_wrong_y1_donor")
            else:raise AssertionError("wrong y1 donor")

        if (name,t) not in (("native_flip",6),("coordinate_flip",15)):
            continue
        def reject(label,bad_entry=entry,bad_stored=stored,bad_payload=payload,
                   bad_emitted=emitted):
            try:
                verify_serialized(fixture,batch,raw,pad,a,old_a,config,bad_emitted,
                                  bad_entry,bad_stored,bad_payload)
            except ValueError:
                checks.append("reject_"+label)
            else:
                raise AssertionError("actual reader accepted "+label)
        changed = {**stored,"input_ids":stored["input_ids"].clone()}
        changed["input_ids"][TARGET,Y1] += 1
        reject("dropped_or_replaced_y1",bad_stored=changed)
        for label,key,index in (("history","input_ids",(TARGET,PROMPT)),
                                ("companion","input_ids",(0,PROMPT)),
                                ("position","position_ids",(0,TARGET,Y1))):
            bad = {**stored,key:stored[key].clone()};bad[key][index] += 1
            reject(label,bad_stored=bad)
        bad = {**payload,"actual_mask":payload["actual_mask"].clone()}
        bad["actual_mask"][0,0,-1,0] = ~bad["actual_mask"][0,0,-1,0]
        reject("companion_complement_mask",bad_payload=bad)
        bad = {**payload,"actual_mask":payload["actual_mask"].clone()}
        bad["actual_mask"][TARGET,0,-1,-1] = False
        reject("current_to_current_mask",bad_payload=bad)
        if config["mask_mode"] == "coordinate":
            bad = {**payload,"actual_mask":native.clone()}
            reject("coordinate_rectangle_missing",bad_payload=bad)
        for label,key,value in (("wrong_arm","arm", "native_sham"),
                                ("wrong_step","step",t+1)):
            reject(label,bad_payload={**payload,key:value})
        bad = {**entry,"raw_argmax":selected if selected != greedy else greedy+1}
        reject("raw_argmax_alias",bad_entry=bad)
        bad = {**entry,"selected":greedy if selected != greedy else greedy+1}
        reject("selected_alias",bad_entry=bad)
        bad = {**entry,"policy":"flip_write"}
        reject("extra_policy_write",bad_entry=bad)
        bad = list(emitted);bad[-1] += 1
        reject("wrong_own_prefix",bad_emitted=bad)
        for label,kw in (("wrong_target",{"target":1}),
                         ("wrong_history_keys",{"key":(PROMPT+1,PROMPT+OFFSET)}),
                         ("wrong_query",{"query":(PROMPT+OFFSET-1,PROMPT+OFFSET+t)}),
                         ("wrong_stage_split",{"split":prior.SPLIT+1})):
            try:prior.mask_for(full,old_a,mode,**kw)
            except ValueError:checks.append("reject_"+label)
            else:raise AssertionError(label)
    sample = [{"arm":x,"mask_mode":spec(a,x)["mask_mode"],
               "override_token":spec(a,x)["override_token"],
               "steps":[{"step":0}]} for x in ARMS]
    receipt_guard(sample,a)
    for bad in (sample[:-1],sample[1:2]+sample[:1]+sample[2:]):
        try:receipt_guard(bad,a)
        except ValueError:checks.append("reject_missing_or_reordered_arm")
        else:raise AssertionError("arm order")
    return checks


def preflight():
    a,old_a = contract()
    require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),
            "attempt already prepared")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preparation loaded model")
    batch,raw,trace,sr,planning = prior.source(q,old_a,torch.device("cpu"))
    pad = int(q.tokenizer.pad_token_id);special = frozenset(q.tokenizer.all_special_ids)
    checks = cpu_checks(a,old_a,batch,raw,pad,special)
    require(len(checks) >= 20,"actual caller/reader mutation checks missing")
    fixture = base.ConfigOnlyRope();widths = []
    for t in range(16):
        emitted = list(prior.old.ROW1[:t]) if t <= 9 else [151646]+[8987]*(t-1)
        emitted[5:6] = [151683] if t > 5 else []
        full = step_inputs(fixture,batch,raw,pad,spec(a,"native_flip"),emitted,
                           previous=emitted[-1] if t else None)
        widths.append(int(full["input_ids"].shape[1]))
    pixels = int(batch.inputs["pixel_values"].numel())
    require(widths == list(range(1371,1387)) and pixels == 24502272 and
            [len(x["token_ids"]) for x in raw] == [255,37,3084,3084] and
            list(batch.request_ids) == a["source"]["request_ids"],
            "source shape/batch changed")
    old_outer = json.loads(Path(a["predecessor_outer"]["path"]).read_text())
    old_count = json.loads(Path(a["predecessor_receipt"]["path"]).read_text())["counts"]
    old_sq = 10*sum((1371+t)**2 for t in range(9))
    new_sq = 4*sum((1371+t)**2 for t in range(9))+2*sum(x*x for x in widths)
    forecast = 2*old_outer["outer_seconds"]*new_sq/old_sq
    artifact = 2*a["planning"]["basis"]["accepted_total_output_bytes"]*68/90+64*1024**2
    require(old_outer["outer_seconds"] == 247.983433149755 and
            old_count["model_forwards"] == 90 and
            abs(forecast-a["planning"]["maximum"]["two_x_outer_seconds"]) < 1e-9 and
            abs(artifact-a["planning"]["maximum"]["two_x_plus_64MiB_bytes"]) < 1e-6 and
            artifact < a["planning"]["artifact_planning_envelope_bytes"],
            "shape-aware cost/artifact capacity conflict")
    previous = json.loads(prior.PREFLIGHT.read_text())
    direct = [Path(__file__),Path(prior.__file__),Path(prior.old.__file__),Path(base.__file__),
              Path(inspect.getfile(parse_row)),Path(inspect.getfile(create_causal_mask)),
              Path(inspect.getfile(modeling_qwen3_vl)),Path(inspect.getfile(iou_xyxy))]
    direct += [Path(x["maintained"]["path"]) for x in previous["direct_source_captures"]]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved = base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command = ["python","-B","-m","probes.training_set_completion.recurrence_y1_row_routing.run"]
    packet = {"status":"cpu_qualified_before_gpu","protocol":bind(PROTOCOL),
              "admission":bind(ADMISSION),"producer":bind(Path(__file__)),
              "source_identity":sr["identity"],"source_input_identity":sr["input_identity"],
              "request_ids":list(batch.request_ids),"prompt_lengths":list(map(len,batch.prompt_token_ids)),
              "raw_lengths":[len(x["token_ids"]) for x in raw],"pad_id":pad,
              "special_ids":sorted(special),"pixel_elements":pixels,"widths":widths,
              "image_grids":[list(x) for x in batch.image_grids],
              "cpu_checks":checks,"forecast_outer_seconds":forecast,
              "artifact_forecast_bytes":artifact,"direct_source_captures":captures,
              "commands":{k:command+[k] for k in ("preflight","launch","run","readback")}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),
                      "captures":len(captures),"forecast_outer_seconds":forecast,
                      "artifact_forecast_bytes":artifact}))


def checked():
    a,old_a = contract();p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_gpu" and
            p["protocol"] == bind(PROTOCOL) and p["admission"] == bind(ADMISSION) and
            p["producer"] == bind(Path(__file__)) and
            p["widths"] == list(range(1371,1387)) and
            p["pixel_elements"] == 24502272 and
            p["artifact_forecast_bytes"] < a["planning"]["artifact_planning_envelope_bytes"],
            "frozen preflight/producer/shape differs")
    for item in p["direct_source_captures"]:
        require(bind(item["maintained"]["path"]) == item["maintained"] and
                bind(item["capture"]["path"]) == item["capture"],
                "direct maintained source/capture changed")
    return a,old_a,p


def launch():
    checked()
    require(not (OUT/"outer.json").exists() and not (OUT/"launch.json").exists(),
            "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,"-B","-m",
               "probes.training_set_completion.recurrence_y1_row_routing.run","run"]
    begun = time.monotonic()
    with (OUT/"stdout.log").open("x") as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        code = child.wait()
    packet = {"command":command,"child_pid":child.pid,
              "outer_seconds":time.monotonic()-begun,"returncode":code,"terminal":True}
    write_new(OUT/"outer.json",packet)
    print(json.dumps(packet))
    require(code == 0,"GPU child failed; no retry")


def run():
    a,old_a,p = checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
            "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    started = time.monotonic();device = torch.device("cuda:0");handles = []
    counts = {"model_forwards":0,"vision_forwards":0,"logical_emitted_tokens":0,
              "raw_greedy_selections":0,"identity_y1_writes":0,
              "flip_y1_writes":0,"reused_calls":0}
    receipt = {"status":"running","pid":os.getpid(),"begun_unix":time.time(),
               "protocol":bind(PROTOCOL),"admission":bind(ADMISSION),
               "preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
               "counts":counts,"arms":[]}
    write_new(OUT/"launch.json",receipt)
    active = {}
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q,identity = base.load_model("untied",device)
        expected = p["source_identity"]
        require({k:v for k,v in identity.items() if k != "loader_source"} ==
                {k:v for k,v in expected.items() if k != "loader_source"} and
                all(identity["loader_source"][k] == expected["loader_source"][k]
                    for k in ("sha256","size_bytes")) and
                identity["loader_source"]["path"] == a["loader_crosswalk"]["maintained"]["path"],
                "effective checkpoint/loader differs")
        model = q.model.eval()
        batch,raw,trace,sr,planning = prior.source(q,old_a,device)
        require(sr["input_identity"] == p["source_input_identity"] and
                int(q.tokenizer.pad_token_id) == p["pad_id"] and
                sorted(q.tokenizer.all_special_ids) == p["special_ids"],
                "GPU original source/tokenizer differs")
        pad = p["pad_id"];special = frozenset(p["special_ids"])
        layers = list(model.model.language_model.layers)
        attentions = [x.self_attn for x in layers]
        require(len(layers) == len(attentions) == 28 and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual 28 text layers changed")
        def top(_m,_args,kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 68,"model-forward cap")
            active["actual_input"] = prior.input_hashes(kwargs,kwargs["attention_mask"])
            require(active["actual_input"] == active["expected_input_hashes"],
                    "top-level full input changed")
        def vision(_m,_args):
            counts["vision_forwards"] += 1
            require(counts["vision_forwards"] <= 68,"vision-forward cap")
        handles += [model.register_forward_pre_hook(top,with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(vision)]
        for i,x in enumerate(attentions):
            def hook(_m,_args,kwargs,layer=i):
                mask = kwargs.get("attention_mask")
                require(isinstance(mask,torch.Tensor) and mask.ndim == 4 and
                        torch.equal(mask,active["expected_mask"]),
                        f"layer {layer} consumed wrong mask")
                active["layers"].append(layer)
                active["layer_mask_hashes"].append(base.tensor_hash(mask))
            handles.append(x.register_forward_pre_hook(hook,with_kwargs=True))
        for i,x in enumerate(layers):
            def state_hook(_m,_args,output,layer=i):
                value = output[0] if isinstance(output,tuple) else output
                require(isinstance(value,torch.Tensor) and value.ndim == 3,
                        "layer states unavailable")
                active["history"].append(value[TARGET,PROMPT:PROMPT+OFFSET].detach().cpu().float())
                active["companions"].append(value[[0,1,3],-1].detach().cpu().float())
            handles.append(x.register_forward_hook(state_hook))
        receipt["effective_identity"] = identity
        baselines = {}
        with torch.inference_mode():
            for config in a["arms"]:
                name = config["name"];emitted = []
                row = {"arm":name,"mask_mode":config["mask_mode"],
                       "override_token":config["override_token"],"steps":[],"stop":None}
                receipt["arms"].append(row)
                if name == "native_flip":
                    require(len(receipt["arms"]) == 5 and
                            all(x["stop"] == "complete" and len(x["steps"]) == 9
                                for x in receipt["arms"][:4]) and
                            counts["identity_y1_writes"] == 2,
                            "four nine-step controls not qualified")
                for t in range(config["max_tokens"]):
                    previous = emitted[-1] if t else None
                    full = step_inputs(model,batch,raw,pad,config,emitted,previous=previous)
                    mask,native,blocked = prior.mask_for(full,old_a,alias(config))
                    actual = native if config["mask_mode"] == "native" else mask
                    active.update(full=full,expected_mask=actual,
                                  expected_input_hashes=prior.input_hashes(full,mask))
                    out = prior.caller(model,full,old_a,alias(config),active)
                    torch.cuda.synchronize(device)
                    require(len(active["layers"]) == len(active["history"]) ==
                            len(active["companions"]) == 28 and
                            active.get("actual_input") is not None,
                            "actual consumers/states incomplete")
                    logits = out.logits[:,-1,:].detach().cpu().float()
                    require(logits.shape == (4,152670) and torch.isfinite(logits).all().item(),
                            "invalid full-vocabulary vectors")
                    greedy,selected,policy = select_token(a,config,t,emitted,logits)
                    verify_online(model,batch,raw,pad,a,old_a,config,emitted,full,active,
                                  logits,greedy,selected,policy)
                    payload = {"arm":name,"step":t,"logits":logits,
                               "actual_mask":actual.detach().cpu(),
                               "historical_by_layer":torch.stack(active["history"]),
                               "companions_by_layer":torch.stack(active["companions"])}
                    rawpath = OUT/f"{name}-step{t}.pt";torch.save(payload,rawpath)
                    inputpath = OUT/f"inputs-{name}-step{t}.json"
                    write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                    entry = entry_for(full,mask,native,blocked,active,greedy,selected,policy,
                                      t,rawpath,inputpath,previous)
                    row["steps"].append(entry)
                    stored = {k:torch.tensor(v,dtype=torch.long) for k,v in
                              json.loads(inputpath.read_text()).items()}
                    verify_serialized(model,batch,raw,pad,a,old_a,config,emitted,
                                      entry,stored,payload)
                    entry["accepted_reference_max_error"] = reference(a,config,t,full,logits,greedy)
                    compare_states(config,t,payload,logits,baselines)
                    entry["source_trace_parity"] = trace_check(logits,trace,raw,config,t)
                    emitted.append(selected)
                    counts["logical_emitted_tokens"] += 1
                    counts["raw_greedy_selections"] += 1
                    counts["identity_y1_writes"] += policy == "identity_write"
                    counts["flip_y1_writes"] += policy == "flip_write"
                    require(counts["logical_emitted_tokens"] <= 68,"emitted-token cap")
                    parsed = parse_row(emitted,special)
                    row["emitted"] = list(emitted);row["stop"] = parsed["stop"]
                    receipt["counts"] = dict(counts)
                    write_new(OUT/f"checkpoint-{name}-{t}.json",receipt)
                    if parsed["stop"] is not None:
                        break
                require(row["stop"] is not None,"trajectory did not stop within cap")
                if name in ARMS[:4]:
                    expected_row = [x["chosen"] for x in a["references"][config["reference_arm"]]]
                    require(row["stop"] == "complete" and row["emitted"] == expected_row and
                            len(row["steps"]) == 9,"known same-mode control qualification failed")
                if name in ("native_A","coordinate_A"):
                    baselines[name] = [torch.load(OUT/f"{name}-step{t}.pt",map_location="cpu",
                                                  weights_only=True) for t in range(9)]
        require(counts["identity_y1_writes"] == counts["flip_y1_writes"] == 2,
                "four fixed policy writes incomplete")
        receipt["status"] = "candidate_raw_complete"
    except BaseException as exc:
        receipt["status"] = "technical_invalid"
        receipt["failure"] = {"type":type(exc).__name__,"message":str(exc),
                              "traceback":traceback.format_exc()}
    finally:
        for h in handles:h.remove()
        if torch.cuda.is_available():torch.cuda.synchronize(device)
        receipt["counts"] = dict(counts)
        receipt["internal_seconds"] = time.monotonic()-started
        receipt["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["artifact_bytes"] = sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"] = os.getpid()
        write_new(OUT/"receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"counts":counts,
                      "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"] == "candidate_raw_complete","technical failure; no retry")


def readback():
    a,old_a,p = checked()
    require(not (OUT/"readback.json").exists(),"cold readback already exists")
    receipt = json.loads((OUT/"receipt.json").read_text())
    outer = json.loads((OUT/"outer.json").read_text())
    receipt_guard(receipt["arms"],a)
    require(receipt["status"] == "candidate_raw_complete" and
            receipt["protocol"] == bind(PROTOCOL) and
            receipt["admission"] == bind(ADMISSION) and
            receipt["preflight"] == bind(PREFLIGHT) and
            receipt["producer"] == bind(Path(__file__)) and
            receipt["counts"]["reused_calls"] == 0 and
            outer["terminal"] and outer["returncode"] == 0 and
            outer["child_pid"] == receipt["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists(),
            "terminal receipt/order/producer/outer changed")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold reader loaded language model")
    batch,raw,trace,sr,planning = prior.source(q,old_a,torch.device("cpu"))
    require(sr["input_identity"] == p["source_input_identity"] and
            int(q.tokenizer.pad_token_id) == p["pad_id"] and
            sorted(q.tokenizer.all_special_ids) == p["special_ids"],
            "cold source/tokenizer changed")
    fixture = base.ConfigOnlyRope();special = frozenset(p["special_ids"]);pad = p["pad_id"]
    result = {"status":"candidate_cold_readback_passed","protocol":bind(PROTOCOL),
              "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),
              "producer":bind(Path(__file__)),"receipt":bind(OUT/"receipt.json"),
              "outer":bind(OUT/"outer.json"),"arms":[],"counts":receipt["counts"],
              "outer_seconds":outer["outer_seconds"],
              "cumulative_sequence_gpu_hours":a["prior_sequence_gpu_hours"]+
                                             outer["outer_seconds"]/3600}
    baselines = {};calls = 0;writes = {"identity_write":0,"flip_write":0}
    for row,config in zip(receipt["arms"],a["arms"],strict=True):
        emitted = [];summaries = [];native_payloads = []
        for t,entry in enumerate(row["steps"]):
            require(entry["step"] == t and bind(entry["raw"]["path"]) == entry["raw"] and
                    bind(entry["inputs"]["path"]) == entry["inputs"],
                    "cold raw/input binding changed")
            stored = {k:torch.tensor(v,dtype=torch.long) for k,v in
                      json.loads(Path(entry["inputs"]["path"]).read_text()).items()}
            payload = torch.load(entry["raw"]["path"],map_location="cpu",weights_only=True)
            full = verify_serialized(fixture,batch,raw,pad,a,old_a,config,emitted,
                                     entry,stored,payload)
            logits = payload["logits"]
            require(payload["historical_by_layer"].shape[0] ==
                    payload["companions_by_layer"].shape[0] == 28,
                    "cold all-layer states missing")
            compare_states(config,t,payload,logits,baselines)
            reference(a,config,t,full,logits,entry["raw_argmax"])
            trace_check(logits,trace,raw,config,t)
            v = logits[TARGET].double();top = torch.topk(v,2)
            summaries.append({"step":t,"raw_argmax":entry["raw_argmax"],
                              "selected":entry["selected"],"policy":entry["policy"],
                              "selected_cells":entry["selected_cells"],
                              "top2_ids":top.indices.tolist(),"top2_logits":top.values.tolist(),
                              "log_normalizer":float(torch.logsumexp(v,-1)),"raw":entry["raw"]})
            if entry["policy"] in writes:writes[entry["policy"]] += 1
            emitted.append(entry["selected"]);calls += 1
            parsed = parse_row(emitted,special)
            require((parsed["stop"] is None) if t < len(row["steps"])-1 else
                    parsed["stop"] == row["stop"],"cold parser/stop changed")
            if row["arm"] in ("native_A","coordinate_A"):
                native_payloads.append(payload)
        require(emitted == row["emitted"] and len(emitted) <= config["max_tokens"],
                "cold emitted prefix/count changed")
        parsed = parse_row(emitted,special)
        description = q.tokenizer.decode(parsed["description_ids"],skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        box = [x-151670 for x in parsed["box_ids"]] if len(parsed["box_ids"]) == 4 else None
        geometry = ("valid" if box[0] < box[2] and box[1] < box[3] else "invalid") if box else "no_complete_box"
        outcome = {"arm":row["arm"],"emitted":emitted,"stop":row["stop"],
                   "description_ids":parsed["description_ids"],"description":description,
                   "box":box,"geometry":geometry,"steps":summaries}
        outcome["region"] = prior.classify(outcome,old_a)
        result["arms"].append(outcome)
        if row["arm"] in ("native_A","coordinate_A"):
            baselines[row["arm"]] = native_payloads
        if row["arm"] in ARMS[:4]:
            expected = [x["chosen"] for x in a["references"][config["reference_arm"]]]
            require(row["stop"] == "complete" and emitted == expected and len(emitted) == 9,
                    "cold nine-step control differs")
    require(calls == receipt["counts"]["model_forwards"] ==
            receipt["counts"]["vision_forwards"] ==
            receipt["counts"]["logical_emitted_tokens"] <= 68 and
            writes == {"identity_write":2,"flip_write":2} and len(result["arms"]) == 6,
            "cold finite counts/policy writes changed")
    observed = {x["arm"]:x["region"] for x in result["arms"] if x["arm"] in ARMS[4:]}
    result["flip_regions"] = observed
    result["primary_pass"] = observed == a["primary"]["all"]
    result["comparator_pass"] = observed == a["comparator"]["all"]
    if "numerical_HOLD" in observed.values():result["shared_decision"] = "numerical_HOLD"
    elif result["primary_pass"]:result["shared_decision"] = a["primary"]["name"]
    elif result["comparator_pass"]:result["shared_decision"] = a["comparator"]["name"]
    else:result["shared_decision"] = a["other_category"]
    image = Image.open(a["source_bindings"]["image"]["path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for label,box,color in [("A",a["region_rule"]["A"],"#00ee77"),
                            ("F",a["region_rule"]["F"],"#ff5533")]+[
                            (x["arm"],x["box"],col) for x,col in zip(result["arms"][4:],
                            ("#00d9ff","#ee00dd"),strict=True)]:
        if box is None or box[0] >= box[2] or box[1] >= box[3]:continue
        xy = [round(v*(image.width if i%2 == 0 else image.height)/1000)
              for i,v in enumerate(box)]
        draw.rectangle(xy,outline=color,width=4)
        draw.text((xy[0],max(0,xy[1]-14)),label,fill=color)
    image.save(OUT/"overlay.png");result["overlay"] = bind(OUT/"overlay.png")
    write_new(OUT/"readback.json",result)
    print(json.dumps({"status":result["status"],"regions":observed,
                      "decision":result["shared_decision"],"counts":receipt["counts"]}))


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("action",choices=("preflight","launch","run","readback"))
    {"preflight":preflight,"launch":launch,"run":run,"readback":readback}[arg.parse_args().action]()


if __name__ == "__main__":main()
