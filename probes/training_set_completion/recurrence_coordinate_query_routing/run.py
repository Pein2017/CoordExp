"""Six fixed-prefix coordinate-query masks; no generation."""
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
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_history_read_stage import run as prior


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-coordinate-query-routing"
PROTOCOL = UNIT / "unit.md"
ADMISSION = UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-coordinate-query-routing/attempt-001")
SHAS = {PROTOCOL: "cb8abfdcf8266ffb92266542c987e7c2c5d97605e45a41dfbf70b5f847acc4b6",
        ADMISSION: "27c52b8bfe6f27b50c9d5d5dfb4fe1d24f96f0d6240ac3c78d3c0d57b5a95803"}
ARMS = ("native", "both", "native_sham", "both_sham", "earlier_only", "current_only")
KEYS = prior.KEYS
TARGET, PROMPT, WIDTH, TOL = 2, 1362, 1377, 2e-4
base = prior.base
require, bind, write_new = base.require, base.bind, base.write_new


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-finite-coordinate-query-routing" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            [x["name"] for x in a["cells"]] == list(ARMS) and
            [x["blocked_cells_per_layer"] for x in a["cells"]] == [0,18,0,18,9,9] and
            a["conditioning"]["current_prefix"] == [151646,8987,151647,151648,151670,151670] and
            a["conditioning"]["prefix_width"] == WIDTH and
            a["mask_geometry"]["history_keys_physical"] == [1362,1371] and
            a["mask_geometry"]["earlier_query_physical"] == [1375,1376] and
            a["mask_geometry"]["current_query_physical"] == [1376,1377] and
            a["counts"]["model_forwards"] == a["counts"]["vision_forwards"] == 6 and
            a["counts"]["generated_tokens"] == a["counts"]["reused_calls"] == 0 and
            a["planning"]["artifact_envelope_bytes"] == 256*1024**2 and
            a["owned_paths"]["raw"] == str(OUT), "admitted scope changed")
    for name in ("protocol", "lead_geometry", "predecessor_acceptance", "predecessor_candidate",
                 "predecessor_verification", "predecessor_receipt", "predecessor_readback",
                 "predecessor_outer", "predecessor_producer", "source_panel"):
        require(bind(a[name]["path"]) == a[name], f"binding changed: {name}")
    for name in ("raw", "trace", "runtime_receipt", "image"):
        require(bind(a["source_bindings"][name]["path"]) == a["source_bindings"][name],
                f"source binding changed: {name}")
    cross = a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"]) == cross["maintained"] and
            cross["maintained"]["sha256"] == cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"]) == cross["prior_acceptance"],
            "maintained loader crosswalk changed")
    for key in ("native", "both"):
        for kind in ("raw", "inputs"):
            item = a["references"][key][kind]
            require(bind(item["path"]) == item, f"distinct {key} {kind} reference changed")
    require(a["references"]["native"]["inputs"]["path"] !=
            a["references"]["both"]["inputs"]["path"] and
            a["references"]["native"]["inputs"]["sha256"] ==
            a["references"]["both"]["inputs"]["sha256"],
            "same-prefix reference path/byte identity changed")
    old_a, _, _ = prior.contract()
    require(a["source_bindings"]["raw"] == old_a["source_bindings"]["raw"] and
            a["source_bindings"]["trace"] == old_a["source_bindings"]["trace"] and
            a["source_bindings"]["source_identity"] == old_a["source_bindings"]["source_identity"] and
            a["source"]["A_tokens"] == list(prior.old.ROW0), "original source changed")
    return a, old_a


def spec(a, arm):
    require(arm in ARMS, "unknown cell")
    return a["cells"][ARMS.index(arm)]


def full_input(model, batch, raw, pad, a, *, target=TARGET, prefix=None):
    prefix = a["conditioning"]["current_prefix"] if prefix is None else prefix
    require(target == TARGET and prefix == a["conditioning"]["current_prefix"],
            "target/current prefix changed")
    tails = base._prefix_tokens(raw,15,pad)
    tails[TARGET] = list(prior.old.ROW0) + list(prefix)
    histories = [list(prompt)+tail for prompt,tail in
                 zip(batch.prompt_token_ids,tails,strict=True)]
    full = base.exact_history_inputs(model,batch.inputs,histories,
                                     pad_token_id=pad,logits_to_keep=1)
    full["cache_position"] = torch.arange(WIDTH,device=full["input_ids"].device)
    require(full["input_ids"].shape == full["attention_mask"].shape == (4,WIDTH) and
            full["position_ids"].shape == (3,4,WIDTH) and
            full["input_ids"][:,PROMPT:].tolist() == tails and
            all(tails[i] == raw[i]["token_ids"][:15] for i in (0,1,3)) and
            all(int(full["attention_mask"][i,:PROMPT].sum()) ==
                len(batch.prompt_token_ids[i]) for i in range(4)) and
            full["attention_mask"][:,PROMPT:].all().item() and
            full["position_ids"][:,TARGET,1375].tolist() == [400]*3 and
            full["position_ids"][:,TARGET,1376].tolist() == [401]*3,
            "original full-prefix/source/position changed")
    return full


def mask_for(full, a, arm, *, target=TARGET, key=(1362,1371),
             spans=None, source_hash=None):
    cfg = spec(a,arm)
    wanted = cfg["blocked_query_spans_physical"]
    require(target == TARGET and key == (1362,1371) and
            (spans is None or spans == wanted) and
            (source_hash is None or source_hash == base.tensor_hash(full["attention_mask"])),
            "target/key/query/source mask changed")
    native = base.native_4d(full["attention_mask"])
    selected = torch.zeros_like(native)
    for lo,hi in wanted:
        require([lo,hi] in ([1375,1376],[1376,1377]) and hi-lo == 1,
                "query rectangle changed")
        selected[TARGET,0,lo:hi,key[0]:key[1]] = True
    count = cfg["blocked_cells_per_layer"]
    require(int(selected.sum()) == count and
            (not count or bool(native[selected].all())), "selected edge count/readability changed")
    actual = native.clone();actual[selected] = False
    prior.verify_rectangle(native,actual,selected,count)
    return full["attention_mask"] if arm in ("native","native_sham") else actual,native,selected


def caller(model,full,a,arm,seen):
    mask,native,selected = mask_for(full,a,arm,
                                    source_hash=base.tensor_hash(full["attention_mask"]))
    actual = native if arm in ("native","native_sham") else mask
    seen.update(expected_mask=actual,native_mask=native,selected=selected,
                actual_input=None,layers=[],layer_mask_hashes=[],history=[],current=[],companions=[])
    return model(**{**full,"attention_mask":mask})


def verify_serialized(model,batch,raw,pad,a,arm,entry,stored,payload):
    require(entry["step"] == 6 and entry["arm"] == payload["arm"] == arm and
            entry["order"] == ARMS.index(arm), "serialized cell/order changed")
    full = full_input(model,batch,raw,pad,a)
    mask,native,selected = mask_for(full,a,arm)
    actual = native if arm in ("native","native_sham") else mask
    require(all(torch.equal(stored[k].detach().cpu(),full[k].detach().cpu()) for k in KEYS) and
            entry["input_hashes"] == prior.input_hashes(full,mask) and
            torch.equal(payload["actual_mask"].detach().cpu(),actual.detach().cpu()) and
            entry["actual_mask_hash"] == base.tensor_hash(actual) and
            entry["actual_layers"] == list(range(28)) and
            entry["actual_layer_mask_hashes"] == [base.tensor_hash(actual)]*28 and
            entry["selected_cells"] == int(selected.sum()) and
            entry["selected_native_hash"] == base.tensor_hash(native[selected]) and
            entry["selected_actual_hash"] == base.tensor_hash(actual[selected]) and
            entry["complement_native_hash"] == entry["complement_actual_hash"] ==
                base.tensor_hash(native[~selected]) and
            payload["logits"].shape == (4,152670) and
            torch.isfinite(payload["logits"]).all().item() and
            payload["historical_by_layer"].shape[:2] == (28,9) and
            payload["current_by_layer"].shape[:2] == (28,6) and
            payload["companions_by_layer"].shape[:2] == (28,3),
            "serialized source/mask/consumer/vector/state changed")
    return full


def reference(a,arm,full,logits):
    key = spec(a,arm)["reference"]
    if key is None:return None
    item = a["references"][key]
    saved = json.loads(Path(item["inputs"]["path"]).read_text())
    require(all(saved[k] == full[k].detach().cpu().tolist() for k in KEYS),
            f"{key} distinct accepted input differs")
    old_logits = torch.load(item["raw"]["path"],map_location="cpu",weights_only=True)["logits"]
    error = float((logits-old_logits).abs().max())
    require(error <= TOL,"accepted native/both full-vector reference differs")
    return error


def state_check(arm,payload,logits,baselines):
    native = baselines.get("native")
    if native is None:return
    def err(x,y):return float((x-y).abs().max())
    require(err(payload["historical_by_layer"],native["historical_by_layer"]) <= TOL and
            err(payload["current_by_layer"][:,:4],native["current_by_layer"][:,:4]) <= TOL and
            err(payload["companions_by_layer"],native["companions_by_layer"]) <= TOL and
            max(err(logits[i],native["logits"][i]) for i in (0,1,3)) <= TOL,
            "historical/header/companion state changed")
    if arm == "current_only":
        require(err(payload["current_by_layer"][:,4],native["current_by_layer"][:,4]) <= TOL,
                "current-only changed earlier x1 state")
    if arm in ("native_sham","both_sham"):
        own = baselines["native" if arm == "native_sham" else "both"]
        require(torch.equal(logits,own["logits"]) and
                all(torch.equal(payload[k],own[k]) for k in
                    ("historical_by_layer","current_by_layer","companions_by_layer")),
                "independent identity-mask sham differs")


def trace_check(logits,trace,raw,arm):
    return prior.trace_check(logits,trace,raw,
                             "native_A" if arm in ("native","native_sham") else "coordinate_A",6)


def cpu_checks(a,batch,raw,pad):
    fixture = base.ConfigOnlyRope();cfg = AutoConfig.from_pretrained(base.BASE,local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    full = full_input(fixture,batch,raw,pad,a)
    native = base.native_4d(full["attention_mask"])
    installed = create_causal_mask(cfg,torch.empty((4,WIDTH,1)),full["attention_mask"],
                                   full["cache_position"],None,position_ids=full["position_ids"][0])
    require(torch.equal(native,installed),"actual native SDPA route differs")
    checks = ["native_SDPA_full_prefix"]
    seen_masks = {}
    for arm in ARMS:
        mask,native,selected = mask_for(full,a,arm)
        seen_masks[arm] = native if arm in ("native","native_sham") else mask
        seen = {}
        class Fake:
            def __call__(self,**kwargs):
                seen["actual_input"] = prior.input_hashes(kwargs,kwargs["attention_mask"])
                z = torch.zeros((4,1,152670));z[TARGET,0,151703] = 1
                return SimpleNamespace(logits=z)
        logits = caller(Fake(),full,a,arm,seen).logits[:,-1]
        seen["layers"] = list(range(28))
        seen["layer_mask_hashes"] = [base.tensor_hash(seen["expected_mask"])]*28
        require(seen["actual_input"] == prior.input_hashes(full,mask) and
                seen["layers"] == list(range(28)),"actual CPU caller not exercised")
        entry = prior.entry_for(full,mask,native,selected,seen,151703,6,Path(__file__),Path(__file__))
        entry.update(arm=arm,order=ARMS.index(arm))
        payload = {"arm":arm,"actual_mask":seen["expected_mask"],"logits":logits,
                   "historical_by_layer":torch.zeros((28,9,1)),
                   "current_by_layer":torch.zeros((28,6,1)),
                   "companions_by_layer":torch.zeros((28,3,1))}
        stored = {k:full[k].clone() for k in KEYS}
        verify_serialized(fixture,batch,raw,pad,a,arm,entry,stored,payload)
        checks.append(f"actual_caller_reader_{arm}")
        for label,changed in (("target",{"target":3}),("history_key",{"key":(1363,1371)}),
                              ("wrong_query",{"spans":[[1374,1375]]}),
                              ("swapped_E_C",{"spans":[[1376,1377]] if arm=="earlier_only" else
                                                    [[1375,1376]]} if arm in ("earlier_only","current_only") else
                                                    {"spans":[[1375,1376]]})):
            try:mask_for(full,a,arm,**changed)
            except ValueError:checks.append("reject_"+label+"_"+arm)
            else:raise AssertionError(label)
        def reject(label,entry2=entry,stored2=stored,payload2=payload):
            try:verify_serialized(fixture,batch,raw,pad,a,arm,entry2,stored2,payload2)
            except ValueError:checks.append("reject_"+label+"_"+arm)
            else:raise AssertionError(label)
        wrong = payload["actual_mask"].clone();wrong[TARGET,0,1376,1362] = ~wrong[TARGET,0,1376,1362]
        reject("mask_selected_or_complement",payload2={**payload,"actual_mask":wrong})
        wrong = payload["actual_mask"].clone();wrong[TARGET,0,1376,1376] = False
        reject("current_to_current",payload2={**payload,"actual_mask":wrong})
        for label,key,index in (("history","input_ids",(TARGET,1362)),
                                ("current_prefix","input_ids",(TARGET,1376)),
                                ("companion","input_ids",(0,1376)),
                                ("position","position_ids",(0,TARGET,1376)),
                                ("source_mask","attention_mask",(0,1376))):
            changed = {**stored,key:stored[key].clone()};changed[key][index] += 1
            reject(label,stored2=changed)
        wrongentry = {**entry,"order":(entry["order"]+1)%6};reject("order",entry2=wrongentry)
        wrongentry = {**entry,"selected_cells":entry["selected_cells"]+1};reject("selected",entry2=wrongentry)
        wrongentry = {**entry,"complement_actual_hash":"0"*64};reject("complement",entry2=wrongentry)
    require(torch.equal(seen_masks["native"],seen_masks["native_sham"]) and
            torch.equal(seen_masks["both"],seen_masks["both_sham"]) and
            torch.equal(seen_masks["both"],seen_masks["earlier_only"] & seen_masks["current_only"]) and
            int((seen_masks["native"] & ~seen_masks["both"]).sum()) == 18 and
            int((seen_masks["native"] & ~seen_masks["earlier_only"]).sum()) == 9 and
            int((seen_masks["native"] & ~seen_masks["current_only"]).sum()) == 9,
            "six fixed mask algebra differs")
    checks.append("disjoint_E_C_union")
    return checks


def preflight():
    a,old_a = contract()
    require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),
            "fixed attempt already prepared")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preparation loaded language model")
    batch,raw,trace,sr,planning = prior.source(q,old_a,torch.device("cpu"))
    require(all(sr["input_identity"][k] == v for k,v in
                a["source_bindings"]["input_identity_summary"].items()) and
            list(batch.request_ids) == a["source"]["request_ids"] and
            int(batch.inputs["pixel_values"].numel()) == 24502272 and
            [len(x["token_ids"]) for x in raw] == [255,37,3084,3084],
            "source/shape/input identity changed")
    pad = int(q.tokenizer.pad_token_id)
    checks = cpu_checks(a,batch,raw,pad)
    fixture = base.ConfigOnlyRope()
    full = full_input(fixture,batch,raw,pad,a)
    for key in ("native","both"):
        saved = json.loads(Path(a["references"][key]["inputs"]["path"]).read_text())
        require(all(saved[k] == full[k].tolist() for k in KEYS),
                f"{key} original-path reference input differs")
    require(len(checks)>=60 and full["input_ids"].shape == (4,WIDTH) and
            a["planning"]["artifact_forecast_bytes"] <
            a["planning"]["artifact_envelope_bytes"],
            "CPU caller or capacity forecast failed")
    old_preflight = json.loads(Path(old_a["one_record_receipt"]["path"]).read_text())
    require(old_preflight["counts"]["model_forwards"] == 45,
            "bound older reference receipt changed")
    previous = json.loads((REPO/"research/experiments/2026-09-24-recurrence-y1-row-routing/supporting/attempt-001-preflight.json").read_text())
    direct = [Path(__file__),Path(prior.__file__),Path(prior.old.__file__),Path(base.__file__),
              Path(inspect.getfile(create_causal_mask)),Path(inspect.getfile(modeling_qwen3_vl)),
              Path(inspect.getfile(AutoConfig))]
    direct += [Path(x["maintained"]["path"]) for x in previous["direct_source_captures"]]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else Path("external")/path.name
        saved = base.preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command = ["python","-B","-m",
               "probes.training_set_completion.recurrence_coordinate_query_routing.run"]
    packet = {"status":"cpu_qualified_before_gpu","protocol":bind(PROTOCOL),
              "admission":bind(ADMISSION),"producer":bind(Path(__file__)),
              "source_identity":sr["identity"],"source_input_identity":sr["input_identity"],
              "request_ids":list(batch.request_ids),
              "prompt_lengths":list(map(len,batch.prompt_token_ids)),
              "raw_lengths":[len(x["token_ids"]) for x in raw],
              "pad_id":pad,"special_ids":sorted(q.tokenizer.all_special_ids),
              "pixel_elements":int(batch.inputs["pixel_values"].numel()),
              "width":WIDTH,"image_grids":[list(x) for x in batch.image_grids],
              "cpu_checks":checks,"direct_source_captures":captures,
              "forecast_outer_seconds":a["planning"]["allowance_seconds"],
              "artifact_forecast_bytes":a["planning"]["artifact_forecast_bytes"],
              "commands":{"preflight":command+["preflight"],"launch":command+["launch"],
                          "model":command+["run"],"cold":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"cpu_checks":len(checks),
                      "source_captures":len(captures),"width":WIDTH,
                      "pixels":packet["pixel_elements"],"artifact_forecast_bytes":
                      packet["artifact_forecast_bytes"]}))


def checked():
    a,old_a = contract();p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_gpu" and
            p["protocol"] == bind(PROTOCOL) and p["admission"] == bind(ADMISSION) and
            p["producer"] == bind(Path(__file__)) and p["width"] == WIDTH and
            p["pixel_elements"] == 24502272 and
            p["artifact_forecast_bytes"] < a["planning"]["artifact_envelope_bytes"],
            "frozen CPU preflight/producer differs")
    for item in p["direct_source_captures"]:
        require(bind(item["maintained"]["path"]) == item["maintained"] and
                bind(item["capture"]["path"]) == item["capture"],
                "direct maintained source/capture changed")
    return a,old_a,p


def launch():
    checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"outer.json").exists(),
            "fixed attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,"-B","-m",
               "probes.training_set_completion.recurrence_coordinate_query_routing.run","run"]
    begun = time.monotonic()
    with (OUT/"stdout.log").open("x") as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        code = child.wait()
    packet = {"command":command,"child_pid":child.pid,
              "outer_seconds":time.monotonic()-begun,"returncode":code,"terminal":True}
    write_new(OUT/"outer.json",packet)
    print(json.dumps(packet))
    require(code == 0,"model attempt failed; no retry")


def run():
    a,old_a,p = checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
            "fixed attempt already launched")
    OUT.mkdir(parents=True,exist_ok=True)
    started = time.monotonic();device = torch.device("cuda:0");handles = []
    counts = {"model_forwards":0,"vision_forwards":0,"generated_tokens":0,"reused_calls":0}
    receipt = {"status":"running","pid":os.getpid(),"begun_unix":time.time(),
               "protocol":bind(PROTOCOL),"admission":bind(ADMISSION),
               "preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
               "counts":counts,"cells":[]}
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
                "GPU source/tokenizer differs")
        layers = list(model.model.language_model.layers)
        attentions = [x.self_attn for x in layers]
        require(len(layers) == len(attentions) == 28 and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual all-layer route changed")
        def top(_m,_args,kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 6,"model-forward cap")
            active["actual_input"] = prior.input_hashes(kwargs,kwargs["attention_mask"])
            require(active["actual_input"] == active["expected_input_hashes"],
                    "top-level full input changed")
        def vision(_m,_args):
            counts["vision_forwards"] += 1
            require(counts["vision_forwards"] <= 6,"vision-forward cap")
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
                        "decoder layer states unavailable")
                active["history"].append(value[TARGET,1362:1371].detach().cpu().float())
                active["current"].append(value[TARGET,1371:1377].detach().cpu().float())
                active["companions"].append(value[[0,1,3],-1].detach().cpu().float())
            handles.append(x.register_forward_hook(state_hook))
        receipt["effective_identity"] = identity
        baselines = {}
        with torch.inference_mode():
            for arm in ARMS:
                if arm == "earlier_only":
                    require(len(receipt["cells"]) == 4 and
                            [x["arm"] for x in receipt["cells"]] == list(ARMS[:4]) and
                            counts["model_forwards"] == counts["vision_forwards"] == 4,
                            "four full/reference/sham gates incomplete")
                full = full_input(model,batch,raw,p["pad_id"],a)
                mask,native,selected = mask_for(full,a,arm)
                actual = native if arm in ("native","native_sham") else mask
                active.update(expected_mask=actual,
                              expected_input_hashes=prior.input_hashes(full,mask))
                out = caller(model,full,a,arm,active)
                torch.cuda.synchronize(device)
                require(len(active["layers"]) == len(active["history"]) ==
                        len(active["current"]) == len(active["companions"]) == 28 and
                        active.get("actual_input") is not None,
                        "actual consumers/states incomplete")
                logits = out.logits[:,-1,:].detach().cpu().float()
                require(logits.shape == (4,152670) and torch.isfinite(logits).all().item(),
                        "full-vocabulary vector invalid")
                payload = {"arm":arm,"logits":logits,"actual_mask":actual.detach().cpu(),
                           "historical_by_layer":torch.stack(active["history"]),
                           "current_by_layer":torch.stack(active["current"]),
                           "companions_by_layer":torch.stack(active["companions"])}
                rawpath = OUT/f"{arm}.pt";torch.save(payload,rawpath)
                inputpath = OUT/f"inputs-{arm}.json"
                write_new(inputpath,{k:full[k].detach().cpu().tolist() for k in KEYS})
                entry = prior.entry_for(full,mask,native,selected,active,
                                        int(torch.argmax(logits[TARGET]).item()),6,
                                        rawpath,inputpath)
                entry.update(arm=arm,order=ARMS.index(arm))
                receipt["cells"].append(entry)
                stored = {k:torch.tensor(v,dtype=torch.long) for k,v in
                          json.loads(inputpath.read_text()).items()}
                verify_serialized(model,batch,raw,p["pad_id"],a,arm,entry,stored,payload)
                entry["accepted_reference_max_error"] = reference(a,arm,full,logits)
                state_check(arm,payload,logits,baselines)
                entry["source_trace_parity"] = trace_check(logits,trace,raw,arm)
                if arm in ("native","both"):
                    baselines[arm] = payload
                receipt["counts"] = dict(counts)
                write_new(OUT/f"checkpoint-{arm}.json",receipt)
        require(all(counts[k] == a["counts"][k] for k in counts) and
                len(receipt["cells"]) == 6,
                "fixed six-call ledger incomplete")
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
    require(receipt["status"] == "candidate_raw_complete" and
            receipt["protocol"] == bind(PROTOCOL) and
            receipt["admission"] == bind(ADMISSION) and
            receipt["preflight"] == bind(PREFLIGHT) and
            receipt["producer"] == bind(Path(__file__)) and
            [x["arm"] for x in receipt["cells"]] == list(ARMS) and
            outer["terminal"] and outer["returncode"] == 0 and
            outer["child_pid"] == receipt["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists(),
            "terminal receipt/order/outer changed")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold reader loaded language model")
    batch,raw,trace,sr,planning = prior.source(q,old_a,torch.device("cpu"))
    require(sr["input_identity"] == p["source_input_identity"] and
            int(q.tokenizer.pad_token_id) == p["pad_id"] and
            sorted(q.tokenizer.all_special_ids) == p["special_ids"],
            "cold source/tokenizer differs")
    fixture = base.ConfigOnlyRope();baselines = {};vectors = {};summaries = []
    for arm,entry in zip(ARMS,receipt["cells"],strict=True):
        require(bind(entry["raw"]["path"]) == entry["raw"] and
                bind(entry["inputs"]["path"]) == entry["inputs"],
                "cold raw vector/input changed")
        stored = {k:torch.tensor(v,dtype=torch.long) for k,v in
                  json.loads(Path(entry["inputs"]["path"]).read_text()).items()}
        payload = torch.load(entry["raw"]["path"],map_location="cpu",weights_only=True)
        full = verify_serialized(fixture,batch,raw,p["pad_id"],a,arm,entry,stored,payload)
        logits = payload["logits"]
        require(entry["chosen"] == int(torch.argmax(logits[TARGET]).item()),
                "cold argmax changed")
        ref_error = reference(a,arm,full,logits)
        require(entry["accepted_reference_max_error"] == ref_error,
                "cold reference error changed")
        state_check(arm,payload,logits,baselines)
        require(entry["source_trace_parity"] == trace_check(logits,trace,raw,arm),
                "cold source/companion trace changed")
        if arm in ("native","both"):baselines[arm] = payload
        vectors[arm] = logits[TARGET].double()
        top = torch.topk(vectors[arm],2)
        z = torch.logsumexp(vectors[arm],-1)
        pids = (151703,152206,152669)
        summaries.append({"arm":arm,"raw":entry["raw"],"inputs":entry["inputs"],
                          "selected_cells":entry["selected_cells"],
                          "reference_max_error":ref_error,
                          "winner":int(top.indices[0]),"runner_up":int(top.indices[1]),
                          "top2_logits":top.values.tolist(),
                          "x2_fixed_tokens":{str(k):{"logit":float(vectors[arm][k]),
                                                      "probability":float(torch.exp(vectors[arm][k]-z))}
                                             for k in pids}})
    require(all(receipt["counts"][k] == a["counts"][k] for k in
                ("model_forwards","vision_forwards","generated_tokens","reused_calls")) and
            len(receipt["cells"]) == 6,"cold finite ledger changed")
    probabilities = {k:torch.softmax(v,-1) for k,v in vectors.items()}
    def tv(left,right):
        return float(torch.abs(probabilities[left]-probabilities[right]).sum()/2)
    B = tv("native","both")
    distances = {k:{"to_native":tv(k,"native"),"to_both":tv(k,"both")}
                 for k in ARMS}
    guard = float(a["metrics"]["guard"])
    rE = distances["earlier_only"]["to_both"]/B if B>guard else None
    rC = distances["current_only"]["to_both"]/B if B>guard else None
    if B<=guard:
        primary = comparator = False;decision = a["low_baseline_category"]
    else:
        require(rE is not None and rC is not None,"ratio missing")
        near = any(abs(v)<=guard for v in
                   (rE-0.5,rC-0.5,rE-rC-0.1,rC-rE-0.1))
        primary = rC<0.5 and rE-rC>0.1
        comparator = rE<0.5 and rC-rE>0.1
        decision = (a["numerical_guard_category"] if near else
                    a["primary"]["name"] if primary else
                    a["comparator"]["name"] if comparator else
                    a["other_category"])
    result = {"status":"candidate_cold_readback_passed",
              "protocol":bind(PROTOCOL),"admission":bind(ADMISSION),
              "preflight":bind(PREFLIGHT),"producer":bind(Path(__file__)),
              "receipt":bind(OUT/"receipt.json"),"outer":bind(OUT/"outer.json"),
              "cells":summaries,"counts":receipt["counts"],
              "TV_native_both":B,"TV_distances":distances,
              "r_earlier":rE,"r_current":rC,
              "primary_pass":primary,"comparator_pass":comparator,
              "decision":decision,"outer_seconds":outer["outer_seconds"],
              "cumulative_sequence_gpu_hours":a["prior_sequence_gpu_hours"]+
                                              outer["outer_seconds"]/3600}
    write_new(OUT/"readback.json",result)
    print(json.dumps({"status":result["status"],"B":B,"r_earlier":rE,
                      "r_current":rC,"decision":decision,"counts":receipt["counts"]}))


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("action",choices=("preflight","launch","run","readback"))
    {"preflight":preflight,"launch":launch,"run":run,"readback":readback}[
        arg.parse_args().action]()


if __name__ == "__main__":
    main()
