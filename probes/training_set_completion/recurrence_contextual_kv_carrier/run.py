"""One admitted AF/FF contextual K/V carrier contrast at a common replayed header."""
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
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_book_first_revisit import run as base
from probes.training_set_completion.recurrence_collapse_two_record_routing import run as prior
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest
from src.artifacts.source_provenance import preserve_source


ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "research/experiments/2026-09-24-recurrence-contextual-kv-carrier"
PROTOCOL, ADMISSION = UNIT / "unit.md", UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-contextual-kv-carrier/attempt-001")
SHAS = {PROTOCOL: "630e3faf1f5ffac24fec7a40f8983e482aba5b9c93f91fa8624debce03c2753c",
        ADMISSION: "4b52088233fd8fc5c5e898de31844b990e026fb2b53167c8f192ce103536ac18"}
NAMES = ("full_AF", "full_FF", "prefill_AF", "prefill_FF", "anchor_AF",
         "anchor_FF", "sham_AF", "sham_FF", "hybrid_AF_FF", "hybrid_FF_AF")
ORIGINS = (("AF","AF"),("FF","FF"),("AF","AF"),("FF","FF"),
           ("AF","AF"),("FF","FF"),("AF","AF"),("FF","FF"),
           ("AF","FF"),("FF","AF"))
TARGET, WIDTH, END, LAYERS, HEADS, DIM, TOL = 2, 1380, 1384, 28, 8, 128, 2e-4
OLDER, LATEST = (1362,1371), (1371,1380)
HEADER = [151646,8987,151647,151648]
bind, require, write_new = base.bind, base.require, base.write_new


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen contract changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["status"] == "lead-admitted-ten-calls-after-CPU-qualification" and
            [x["name"] for x in a["cells"]] == list(NAMES) and
            [(x["E"],x["L"]) for x in a["cells"]] == list(ORIGINS) and
            [x["vision"] for x in a["cells"]] == [1]*4+[0]*6 and
            a["max_model_forwards"] == 10 and a["max_vision_forwards"] == 4 and
            a["max_generated_tokens"] == 0 and a["source"]["target_index"] == TARGET and
            a["source"]["common_header_tokens"] == HEADER and
            a["source"]["earlier_physical"] == list(OLDER) and
            a["source"]["latest_physical"] == list(LATEST) and
            a["source"]["suffix_physical"] == [WIDTH,END] and
            a["criteria"]["baseline_minimum"] == 1e-6 and
            a["criteria"]["numerical_boundary_guard"] == 1e-6 and
            a["qualification"]["cached_full_all_four_max_abs"] == TOL and
            a["owned_paths"][-1] == str(OUT), "admitted finite source/criterion changed")
    for key in ("protocol","predecessor_acceptance","predecessor_admission",
                "CPU_feasibility_acceptance","CPU_feasibility_bindings","CPU_feasibility_report",
                "source_panel"):
        require(bind(a[key]["path"]) == a[key], f"bound {key} changed")
    for key in ("raw","trace","runtime_receipt","image"):
        require(bind(a["source_bindings"][key]["path"]) == a["source_bindings"][key],
                f"original {key} changed")
    require(bind(a["loader_crosswalk"]["maintained"]["path"]) ==
            a["loader_crosswalk"]["maintained"], "maintained loader changed")
    for group in ("saved_full_vector_references","saved_full_inputs"):
        for value in a[group].values():require(bind(value["path"]) == value, f"saved {group} changed")
    return a


def make_full(model, batch, raw, pad, origin):
    require(origin in ("AF","FF"), "wrong history origin")
    full = prior.step_inputs(model,batch,raw,pad,"native_AF" if origin=="AF" else "FF",
                             HEADER,previous=HEADER[-1])
    require(full["input_ids"].shape == full["attention_mask"].shape == (4,END) and
            full["position_ids"].shape == (3,4,END) and
            full["input_ids"][TARGET,WIDTH:END].tolist() == HEADER and
            full["position_ids"][:,TARGET,OLDER[0]:OLDER[1]].tolist() ==
                [list(range(387,396))]*3 and
            full["position_ids"][:,TARGET,LATEST[0]:LATEST[1]].tolist() ==
                [list(range(396,405))]*3 and
            full["position_ids"][:,TARGET,WIDTH:END].tolist() ==
                [list(range(405,409))]*3 and
            all(full["attention_mask"][i,OLDER[0]:END].all().item() for i in range(4)),
            "full source geometry/positions/companions changed")
    full = dict(full)
    full.update(use_cache=False,return_dict=True,logits_to_keep=1)
    return full


def split_inputs(full, cache, kind):
    require(kind in ("prefill","suffix") and full["input_ids"].shape == (4,END),
            "wrong cache split")
    if kind == "prefill":
        result = dict(full)
        result.update(input_ids=full["input_ids"][:,:WIDTH],
                      attention_mask=full["attention_mask"][:,:WIDTH],
                      position_ids=full["position_ids"][:,:,:WIDTH],
                      cache_position=torch.arange(WIDTH,device=full["input_ids"].device),
                      past_key_values=cache,use_cache=True)
    else:
        result = {"input_ids":full["input_ids"][:,WIDTH:END],
                  "attention_mask":full["attention_mask"],
                  "position_ids":full["position_ids"][:,:,WIDTH:END],
                  "cache_position":torch.arange(WIDTH,END,device=full["input_ids"].device),
                  "past_key_values":cache,"use_cache":True,"return_dict":True,"logits_to_keep":1}
    require(result["input_ids"].shape == (4,WIDTH if kind=="prefill" else 4) and
            result["position_ids"].shape == (3,4,WIDTH if kind=="prefill" else 4) and
            result["cache_position"].tolist() == list(range(0,WIDTH) if kind=="prefill" else range(WIDTH,END)) and
            torch.equal(result["input_ids"],full["input_ids"][:,:WIDTH] if kind=="prefill" else full["input_ids"][:,WIDTH:END]) and
            torch.equal(result["position_ids"],full["position_ids"][:,:,:WIDTH] if kind=="prefill" else full["position_ids"][:,:,WIDTH:END]) and
            torch.equal(result["attention_mask"],full["attention_mask"][:,:WIDTH] if kind=="prefill" else full["attention_mask"]),
            "source split/companion/position changed")
    return result


def segments(cache, *, width=WIDTH, older=OLDER, latest=LATEST, target=TARGET):
    require(target == TARGET and older[1] == latest[0] and latest[1] == width and
            older[1]-older[0] == latest[1]-latest[0] == 9 and len(cache.layers) == LAYERS,
            "wrong cache segment geometry")
    result=[]
    for layer in cache.layers:
        require(all(getattr(layer,n).shape == (4,HEADS,width,DIM) and
                    getattr(layer,n).dtype == torch.float32 for n in ("keys","values")),
                "historical cache shape/dtype changed")
        result.append({name:{"prompt":base.tensor_hash(t[target,:,:older[0],:]),
                             "older":base.tensor_hash(t[target,:,older[0]:older[1],:]),
                             "latest":base.tensor_hash(t[target,:,latest[0]:latest[1],:]),
                             "companions":base.tensor_hash(t[[0,1,3]])}
                       for name,t in (("keys",layer.keys),("values",layer.values))})
    return result


def blocks(cache, origin, *, older=OLDER, latest=LATEST, target=TARGET):
    require(origin in ("AF","FF") and target == TARGET and len(cache.layers) == LAYERS,
            "wrong block origin/target/layers")
    return {"origin":origin,"layers":[{span:{name:getattr(layer,name)[target,:,a:b,:].detach().cpu().clone()
                                            for name in ("keys","values")}
                                    for span,(a,b) in (("older",older),("latest",latest))}
                                   for layer in cache.layers]}


def verify_selected(cache, base_segments, donor_blocks, *, width=WIDTH,
                    older=OLDER, latest=LATEST, target=TARGET):
    require(target == TARGET and len(base_segments) == len(donor_blocks["layers"]) == LAYERS,
            "wrong selected-cache target/layers")
    observed=segments(cache,width=width,older=older,latest=latest,target=target)
    for i,row in enumerate(observed):
        for name in ("keys","values"):
            for span in ("prompt","older","companions"):
                require(row[name][span] == base_segments[i][name][span],
                        f"layer {i} changed unselected {name} {span}")
            donor=base.tensor_hash(donor_blocks["layers"][i]["latest"][name])
            require(row[name]["latest"] == donor, f"layer {i} consumed wrong latest {name}")
    return observed


@contextmanager
def latest_patch(cache, donor_blocks, *, base_origin, donor_origin, original_digest,
                 width=WIDTH, older=OLDER, latest=LATEST, target=TARGET):
    """Actual suffix caller remains inside inference mode through crop/restore."""
    with torch.inference_mode():
        require(base_origin in ("AF","FF") and donor_origin in ("AF","FF") and
                donor_blocks["origin"] == donor_origin and target == TARGET and
                older[1] == latest[0] and latest[1] == width and
                older[1]-older[0] == latest[1]-latest[0] == 9 and
                len(cache.layers) == len(donor_blocks["layers"]) == LAYERS and
                cache.get_seq_length() == width and cache_digest(cache) == original_digest,
                "wrong patch base/donor/target/span/cache")
        saved=[]
        try:
            for i,layer in enumerate(cache.layers):
                old={}; saved.append((layer,old))
                for name in ("keys","values"):
                    value=getattr(layer,name)
                    donor=donor_blocks["layers"][i]["latest"][name]
                    require(donor.shape == (HEADS,9,DIM) and torch.isfinite(donor).all().item(),
                            f"invalid {name} donor")
                    old[name]=value[target,:,latest[0]:latest[1],:].clone()
                    value[target,:,latest[0]:latest[1],:].copy_(donor.to(value.device))
            yield
        finally:
            try:cache.crop(width)
            finally:
                for layer,old in saved:
                    for name,value in old.items():
                        getattr(layer,name)[target,:,latest[0]:latest[1],:].copy_(value)
            require(cache.get_seq_length() == width and cache_digest(cache) == original_digest,
                    "suffix crop or K/V finally restoration failed")


def record_guard(cells):
    require(isinstance(cells,list) and len(cells)==10, "cold cell list/count changed")
    for i,cell in enumerate(cells):
        require(isinstance(cell,dict) and cell.get("call")==i+1 and
                cell.get("name")==NAMES[i] and cell.get("origins")==list(ORIGINS[i]) and
                isinstance(cell.get("consumer"),dict) and
                isinstance(cell.get("input"),dict),"cold cell order/origin/container changed")


def verify_actual_input(actual, expected, *, media):
    for name in ("input_ids","attention_mask","position_ids","cache_position"):
        require(name in actual and torch.equal(actual[name],expected[name]),
                f"actual {name} differs from source")
    require(actual.get("past_key_values") is expected.get("past_key_values") and
            actual.get("use_cache") is expected["use_cache"] and
            actual.get("logits_to_keep") == 1 and
            all((name in actual)==media and
                (not media or base.tensor_hash(actual[name])==base.tensor_hash(expected[name]))
                for name in ("pixel_values","image_grid_thw")),
            "actual cache/media route differs")


def verify_attention(actual, expected):
    require(isinstance(actual,torch.Tensor) and actual.dtype==torch.bool and
            actual.shape==expected.shape and torch.equal(actual,expected),
            "actual causal/companion attention mask changed")


def expected_suffix_mask(full):
    mask=full["attention_mask"]
    causal=(torch.arange(END,device=mask.device)[None,:] <=
            (WIDTH+torch.arange(4,device=mask.device)[:,None]))
    return (causal[None,None,:,:] & mask[:,None,None,:].bool())


def cpu_patch_checks():
    """Small real DynamicCache fixture; no model, vision, or CUDA."""
    width,older,latest=22,(4,13),(13,22)
    caches={}
    for origin in ("AF","FF"):
        cache=DynamicCache()
        for i in range(LAYERS):
            k=torch.zeros((4,HEADS,width,DIM),dtype=torch.float32)
            v=torch.zeros_like(k)
            for tensor,shift in ((k,0),(v,100)):
                tensor[TARGET,:,older[0]:older[1],:]=i+1+shift+(0 if origin=="AF" else 20)
                tensor[TARGET,:,latest[0]:latest[1],:]=i+2+shift+(0 if origin=="AF" else 30)
            cache.update(k,v,i)
        caches[origin]=cache
    snapshots={o:blocks(c,o,older=older,latest=latest) for o,c in caches.items()}
    initial={o:cache_digest(c) for o,c in caches.items()}
    original={o:segments(c,width=width,older=older,latest=latest) for o,c in caches.items()}
    require(all(original["AF"][i][n][s]==original["FF"][i][n][s]
                for i in range(LAYERS) for n in ("keys","values")
                for s in ("prompt","companions")), "fixture pre-object/companion changed")
    checks=[]
    for base_origin,donor_origin in (("AF","AF"),("FF","FF"),("AF","FF"),("FF","AF")):
        cache=caches[base_origin]
        with latest_patch(cache,snapshots[donor_origin],base_origin=base_origin,
                          donor_origin=donor_origin,original_digest=initial[base_origin],
                          width=width,older=older,latest=latest):
            verify_selected(cache,original[base_origin],snapshots[donor_origin],
                            width=width,older=older,latest=latest)
            for i in range(LAYERS):
                zeros=torch.zeros((4,HEADS,4,DIM),dtype=torch.float32)
                cache.update(zeros,zeros,i)
        require(cache_digest(cache)==initial[base_origin], "normal crop/restore failed")
        checks.append(f"actual_patch_{base_origin}_{donor_origin}_all28_normal_restore")
    try:
        with latest_patch(caches["AF"],snapshots["FF"],base_origin="AF",donor_origin="FF",
                          original_digest=initial["AF"],width=width,older=older,latest=latest):
            for i in range(LAYERS):
                zeros=torch.zeros((4,HEADS,4,DIM),dtype=torch.float32)
                caches["AF"].update(zeros,zeros,i)
            raise RuntimeError("forced body failure")
    except RuntimeError as exc:
        require(str(exc)=="forced body failure", "wrong fixture failure")
    require(cache_digest(caches["AF"])==initial["AF"], "exception restore failed")
    checks.append("forced_body_exception_crop_and_KV_restore")
    for label,fn in (
        ("wrong_base",lambda:latest_patch(caches["AF"],snapshots["FF"],base_origin="AF",
                          donor_origin="FF",original_digest=initial["FF"],width=width,older=older,latest=latest)),
        ("wrong_donor",lambda:latest_patch(caches["AF"],snapshots["FF"],base_origin="AF",
                          donor_origin="AF",original_digest=initial["AF"],width=width,older=older,latest=latest)),
        ("wrong_target",lambda:latest_patch(caches["AF"],snapshots["FF"],base_origin="AF",
                          donor_origin="FF",original_digest=initial["AF"],width=width,older=older,latest=latest,target=1)),
        ("wrong_span",lambda:latest_patch(caches["AF"],snapshots["FF"],base_origin="AF",
                          donor_origin="FF",original_digest=initial["AF"],width=width,older=older,latest=(12,21)))):
        try:
            with fn():pass
        except (ValueError,AssertionError):checks.append(f"reject_{label}")
        else:raise AssertionError(f"fixture accepted {label}")
    with latest_patch(caches["AF"],snapshots["FF"],base_origin="AF",donor_origin="FF",
                      original_digest=initial["AF"],width=width,older=older,latest=latest):
        for axis in ("keys","values"):
            layer=caches["AF"].layers[0]
            current=getattr(layer,axis)
            current[TARGET,:,latest[0]:latest[1],:].copy_(
                snapshots["AF"]["layers"][0]["latest"][axis])
            try:verify_selected(caches["AF"],original["AF"],snapshots["FF"],
                                width=width,older=older,latest=latest)
            except ValueError:checks.append(f"reject_missing_{axis}_edit")
            else:raise AssertionError(f"fixture accepted missing {axis} edit")
            current[TARGET,:,latest[0]:latest[1],:].copy_(
                snapshots["FF"]["layers"][0]["latest"][axis])
        layer=caches["AF"].layers[0]
        layer.values[0,:,0,:].add_(1)
        try:verify_selected(caches["AF"],original["AF"],snapshots["FF"],
                            width=width,older=older,latest=latest)
        except ValueError:checks.append("reject_companion_mutation")
        else:raise AssertionError("fixture accepted companion mutation")
        layer.values[0,:,0,:].sub_(1)
        layer.keys[TARGET,:,older[0],:].add_(1)
        try:verify_selected(caches["AF"],original["AF"],snapshots["FF"],
                            width=width,older=older,latest=latest)
        except ValueError:checks.append("reject_older_mutation")
        else:raise AssertionError("fixture accepted older mutation")
        layer.keys[TARGET,:,older[0],:].sub_(1)
    require(cache_digest(caches["AF"])==initial["AF"], "mutation fixture failed restoration")
    stub={"call":1,"name":NAMES[0],"origins":list(ORIGINS[0]),"consumer":{},"input":{}}
    good=[{**stub,"call":i+1,"name":NAMES[i],"origins":list(ORIGINS[i])}
          for i in range(10)]
    record_guard(json.loads(json.dumps(good)))
    checks.append("serialized_list_guard_green_all10")
    for label,bad in (
        ("swapped",good[:8]+[good[9],good[8]]),
        ("dropped",good[:-1]),
        ("extra",good+[good[-1]]),
        ("wrong_donor",good[:8]+[{**good[8],"origins":["AF","AF"]}]+good[9:]),
        ("wrong_container",{"cells":good})):
        try:record_guard(json.loads(json.dumps(bad)))
        except ValueError:checks.append(f"serialized_guard_reject_{label}")
        else:raise AssertionError(f"serialized guard accepted {label}")
    return checks


def cpu_checks(q,batch,raw,pad,a):
    fixture=base.ConfigOnlyRope()
    fulls={o:make_full(fixture,batch,raw,pad,o) for o in ("AF","FF")}
    checks=cpu_patch_checks()
    for origin,full in fulls.items():
        saved=json.loads(Path(a["saved_full_inputs"]["native_AF" if origin=="AF" else "FF"]["path"]).read_text())
        for name in ("input_ids","attention_mask","position_ids","cache_position"):
            require(full[name].cpu().tolist()==saved[name],f"saved {origin} full input differs")
        vector=torch.load(a["saved_full_vector_references"]["native_AF" if origin=="AF" else "FF"]["path"],
                          map_location="cpu",weights_only=True)["logits"]
        require(vector.shape==(4,152670) and torch.isfinite(vector).all().item(),
                f"saved {origin} vector malformed")
        for kind in ("prefill","suffix"):
            cut=split_inputs(full,object(),kind)
            verify_actual_input(cut,cut,media=kind=="prefill")
        checks.append(f"{origin}_saved_full_source_and_split")
    require(torch.nonzero(fulls["AF"]["input_ids"]!=fulls["FF"]["input_ids"],as_tuple=False).tolist()==
            [[TARGET,1367],[TARGET,1368],[TARGET,1369]] and
            torch.equal(fulls["AF"]["attention_mask"],fulls["FF"]["attention_mask"]) and
            torch.equal(fulls["AF"]["position_ids"],fulls["FF"]["position_ids"]),
            "AF/FF common fork changed")
    checks.append("exact_three_written_input_ids_same_latest_header_positions")
    full=fulls["AF"]; cut=split_inputs(full,object(),"suffix")
    for label,field,idx in (("header","input_ids",(TARGET,0)),
                            ("companion","input_ids",(1,0)),
                            ("position","position_ids",(0,TARGET,0)),
                            ("source_mask","attention_mask",(1,WIDTH))):
        changed=dict(cut);changed[field]=cut[field].clone();changed[field][idx]+=1
        try:verify_actual_input(changed,cut,media=False)
        except ValueError:checks.append(f"reject_suffix_{label}")
        else:raise AssertionError(f"accepted suffix {label} mutation")
    for label,field,value in (("cache_slot","cache_position",WIDTH-1),):
        changed=dict(cut);changed[field]=cut[field].clone();changed[field][0]=value
        try:verify_actual_input(changed,cut,media=False)
        except ValueError:checks.append(f"reject_suffix_{label}")
        else:raise AssertionError(f"accepted suffix {label} mutation")
    expected=expected_suffix_mask(full)
    require(expected.shape==(4,1,4,END),"suffix mask shape changed")
    verify_attention(expected,expected)
    changed=expected.clone();changed[TARGET,0,0,LATEST[0]]=~changed[TARGET,0,0,LATEST[0]]
    try:verify_attention(changed,expected)
    except ValueError:checks.append("reject_actual_attention_mask_mutation")
    else:raise AssertionError("accepted mask mutation")
    return checks,fulls


def preflight():
    a=contract();require(not PREFLIGHT.exists() and not (OUT/"launch.json").exists(),
                         "attempt already prepared")
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"CPU preflight loaded language model")
    batch,raw,trace,sr,planning=prior.source(q,a,torch.device("cpu"))
    pad=int(q.tokenizer.pad_token_id)
    checks,fulls=cpu_checks(q,batch,raw,pad,a)
    require(len(checks)>=23 and int(batch.inputs["pixel_values"].numel())==24502272,
            "CPU caller/source qualification incomplete")
    paths=[Path(__file__),Path(prior.__file__),Path(base.__file__),
           Path(inspect.getfile(DynamicCache)),Path(inspect.getfile(modeling_qwen3_vl)),
           Path(inspect.getfile(sdpa_attention)),Path(preserve_source.__code__.co_filename)]
    old_pre=json.loads(prior.PREFLIGHT.read_text())
    paths += [Path(c["maintained"]["path"]) for c in old_pre["direct_source_captures"]]
    captures=[]
    for path in dict.fromkeys(paths):
        rel=path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("installed")/path.name
        saved=preserve_source(path,run_root=OUT,relative_name=rel)
        captures.append({"maintained":bind(path),"capture":bind(saved)})
    command=["python","-B","-m","probes.training_set_completion.recurrence_contextual_kv_carrier.run"]
    packet={"status":"cpu_qualified_before_gpu","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"producer":bind(Path(__file__)),
            "source_identity":sr["identity"],"request_ids":list(batch.request_ids),
            "input_identity":sr["input_identity"],"pad_id":pad,
            "prompt_lengths":list(map(len,batch.prompt_token_ids)),
            "raw_lengths":[len(x["token_ids"]) for x in raw],
            "pixel_elements":int(batch.inputs["pixel_values"].numel()),
            "shapes":{"full":[4,END],"prefill":[4,WIDTH],"suffix":[4,4],
                      "suffix_mask":[4,1,4,END]},"checks":checks,
            "full_inputs":{o:{name:base.tensor_hash(v) for name,v in full.items()
                              if isinstance(v,torch.Tensor)} for o,full in fulls.items()},
            "source_captures":captures,"forecast_outer_seconds":a["planning_outer_seconds"],
            "artifact_planning_bytes":a["artifact_planning_bytes"],
            "commands":{"preflight":command+["preflight"],"run":command+["run"],
                        "gpu_child":command+["gpu"],"readback":command+["readback"]}}
    write_new(PREFLIGHT,packet)
    print(json.dumps({"status":packet["status"],"checks":len(checks),
                      "captures":len(captures),"shape":packet["shapes"],
                      "forecast_outer_seconds":packet["forecast_outer_seconds"]}))


def checked():
    a=contract();p=json.loads(PREFLIGHT.read_text())
    require(p["status"]=="cpu_qualified_before_gpu" and p["protocol"]==bind(PROTOCOL) and
            p["admission"]==bind(ADMISSION) and p["producer"]==bind(Path(__file__)),
            "preflight/producer binding changed")
    for c in p["source_captures"]:
        require(bind(c["maintained"]["path"])==c["maintained"] and
                bind(c["capture"]["path"])==c["capture"],"direct captured source changed")
    return a,p


def layer_selected(layer, base_row, donor_row, *, target=TARGET):
    require(target==TARGET,"wrong actual target")
    row={}
    for name in ("keys","values"):
        t=getattr(layer,name)
        hashes={"prompt":base.tensor_hash(t[target,:,:OLDER[0],:]),
                "older":base.tensor_hash(t[target,:,OLDER[0]:OLDER[1],:]),
                "latest":base.tensor_hash(t[target,:,LATEST[0]:LATEST[1],:]),
                "companions":base.tensor_hash(t[[0,1,3]])}
        for span in ("prompt","older","companions"):
            require(hashes[span]==base_row[name][span],f"actual unselected {name} {span} changed")
        require(hashes["latest"]==base.tensor_hash(donor_row["latest"][name]),
                f"actual selected latest {name} donor changed")
        row[name]=hashes
    return row


@contextmanager
def suffix_scope(cache, original_digest, *, base_origin, donor_origin=None, donor_blocks=None):
    with torch.inference_mode():
        if donor_origin is None:
            require(cache.get_seq_length()==WIDTH and cache_digest(cache)==original_digest,
                    "native suffix cache changed")
            try:yield
            finally:
                cache.crop(WIDTH)
                require(cache.get_seq_length()==WIDTH and cache_digest(cache)==original_digest,
                        "native suffix crop/restoration failed")
        else:
            require(donor_blocks is not None,"missing K/V donor")
            with latest_patch(cache,donor_blocks,base_origin=base_origin,
                              donor_origin=donor_origin,original_digest=original_digest):
                yield


def gpu_child():
    a,p=checked()
    require(not (OUT/"launch.json").exists() and not (OUT/"receipt.json").exists(),
            "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); device=torch.device("cuda:0"); handles=[]
    counts={"model_forwards":0,"vision_forwards":0,"generated_tokens":0}
    receipt={"status":"running","pid":os.getpid(),"begun_unix":time.time(),
             "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),
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
                "effective model/loader identity changed")
        model=q.model.eval()
        batch,raw,trace,sr,_=prior.source(q,a,device)
        require(sr["input_identity"]==p["input_identity"] and
                int(q.tokenizer.pad_token_id)==p["pad_id"],"GPU source/tokenizer changed")
        pad=p["pad_id"]
        fulls={o:make_full(model,batch,raw,pad,o) for o in ("AF","FF")}
        require({o:{name:base.tensor_hash(v) for name,v in full.items()
                    if isinstance(v,torch.Tensor)} for o,full in fulls.items()}==p["full_inputs"],
                "GPU full source input differs from CPU preflight")
        attentions=[layer.self_attn for layer in model.model.language_model.layers]
        require(len(attentions)==LAYERS and
                all(isinstance(x,modeling_qwen3_vl.Qwen3VLTextAttention) for x in attentions),
                "actual text attention route changed")

        def before_model(_module,_args,kwargs):
            counts["model_forwards"]+=1
            require(counts["model_forwards"]<=10 and active.get("name")==NAMES[counts["model_forwards"]-1],
                    "unexpected model call/order")
            verify_actual_input(kwargs,active["input"],media=active["media"])
            active["top_input"]={k:base.tensor_hash(kwargs[k]) for k in
                                 ("input_ids","attention_mask","position_ids","cache_position")}
        def before_vision(_module,_args):
            counts["vision_forwards"]+=1
            require(active.get("media") and counts["vision_forwards"]<=4,
                    "unexpected vision call")
        def on_rotary(_module,args,output):
            require(len(args)>=2 and torch.equal(args[1],active["input"]["position_ids"]),
                    "actual rotary position input changed")
            cos,sin=output
            n=active["input"]["input_ids"].shape[1]
            require(cos.shape==sin.shape==(4,n,DIM) and not active["rotary_values"],
                    "rotary output shape/count changed")
            active["rotary_values"].append((cos,sin))
            active["rotary_hashes"]={"cos":base.tensor_hash(cos),"sin":base.tensor_hash(sin)}
        handles.extend((model.register_forward_pre_hook(before_model,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision),
                        model.model.language_model.rotary_emb.register_forward_hook(on_rotary)))
        for i,attention in enumerate(attentions):
            def at_entry(_module,_args,kwargs,layer=i):
                mask=kwargs.get("attention_mask")
                verify_attention(mask,active["expected_mask"])
                embed=kwargs.get("position_embeddings")
                require(len(active["rotary_values"])==1 and isinstance(embed,tuple) and len(embed)==2 and
                        torch.equal(embed[0],active["rotary_values"][0][0]) and
                        torch.equal(embed[1],active["rotary_values"][0][1]),
                        "actual attention rotary consumer changed")
                record={"layer":layer,"mask_sha256":base.tensor_hash(mask),
                        "rotary":active["rotary_hashes"]}
                if active["kind"]=="suffix":
                    cache=active["cache"]
                    require(kwargs.get("past_key_values") is cache and
                            cache.get_seq_length(layer)==WIDTH and
                            torch.equal(kwargs.get("cache_position"),active["input"]["cache_position"]),
                            "actual suffix cache slots changed")
                    record["segments"]=layer_selected(cache.layers[layer],
                        active["base_segments"][layer],active["donor_blocks"]["layers"][layer])
                active["attn"].append(record)
            handles.append(attention.register_forward_pre_hook(at_entry,with_kwargs=True))
        for i,layer in enumerate(model.model.language_model.layers):
            def at_output(_module,_args,idx=i):
                if active["kind"]!="suffix":return
                cache=active["cache"]
                require(cache.get_seq_length(idx)==END and
                        {name:base.tensor_hash(getattr(cache.layers[idx],name)[:,:,:WIDTH,:])
                         for name in ("keys","values")}==active["patched_digest"][idx],
                        "historical K/V changed during suffix")
                row={name:base.tensor_hash(getattr(cache.layers[idx],name)[[0,1,3],:,WIDTH:END,:])
                     for name in ("keys","values")}
                active["after"].append({"layer":idx,"companion_suffix":row,
                                        "historical_digest":active["patched_digest"][idx]})
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(at_output))

        def invoke(name, inputs, *, media, cache=None, base_origin=None, donor_blocks=None,
                   base_segments=None, original_digest=None, donor_origin=None):
            i=NAMES.index(name)
            kind="suffix" if i>=4 else "prefill" if i>=2 else "full"
            expected_mask=(expected_suffix_mask(fulls[base_origin]) if kind=="suffix" else
                           base.native_4d(inputs["attention_mask"]))
            active.clear();active.update(name=name,kind=kind,input=inputs,media=media,
                                         expected_mask=expected_mask,cache=cache,
                                         base_segments=base_segments,donor_blocks=donor_blocks,
                                         attn=[],after=[],rotary_values=[],rotary_hashes={})
            if kind=="suffix":
                with suffix_scope(cache,original_digest,base_origin=base_origin,
                                  donor_origin=donor_origin,donor_blocks=donor_blocks if donor_origin else None):
                    active["patched_digest"]=cache_digest(cache)
                    verify_selected(cache,base_segments,donor_blocks)
                    with torch.inference_mode():logits=model(**inputs).logits[:,-1,:].detach().float().cpu()
            else:
                with torch.inference_mode():logits=model(**inputs).logits[:,-1,:].detach().float().cpu()
            torch.cuda.synchronize(device)
            require(logits.shape==(4,152670) and torch.isfinite(logits).all().item() and
                    [x["layer"] for x in active["attn"]]==list(range(LAYERS)) and
                    len(active["rotary_values"])==1 and
                    (kind!="suffix" or [x["layer"] for x in active["after"]]==list(range(LAYERS))),
                    f"{name} actual consumer/vectors incomplete")
            cell_dir=OUT/"cells"/f"{i+1:02d}-{name}";cell_dir.mkdir(parents=True,exist_ok=False)
            torch.save(logits,cell_dir/"full-batch-vocabulary.pt")
            write_new(cell_dir/"input.json",{k:v.detach().cpu().tolist() for k,v in inputs.items()
                                                  if k in ("input_ids","attention_mask","position_ids","cache_position")})
            consumer={"name":name,"kind":kind,"actual_input":active["top_input"],
                      "rotary":active["rotary_hashes"],"attention":active["attn"],
                      "after_suffix":active["after"],
                      "patched_digest":active.get("patched_digest"),
                      "restored_digest":cache_digest(cache) if kind=="suffix" else None,
                      "expected_mask_sha256":base.tensor_hash(expected_mask)}
            write_new(cell_dir/"consumer.json",consumer)
            cell={"call":i+1,"name":name,"origins":list(ORIGINS[i]),
                  "input":bind(cell_dir/"input.json"),
                  "vector":bind(cell_dir/"full-batch-vocabulary.pt"),
                  "consumer":bind(cell_dir/"consumer.json"),
                  "counts_after":dict(counts)}
            receipt["cells"].append(cell)
            return logits,consumer

        reference={}
        for name,origin in (("full_AF","AF"),("full_FF","FF")):
            logits,_=invoke(name,fulls[origin],media=True)
            saved=torch.load(a["saved_full_vector_references"]["native_AF" if origin=="AF" else "FF"]["path"],
                             map_location="cpu",weights_only=True)["logits"]
            err=float((logits-saved).abs().max())
            require(err<=TOL,f"{name} saved all-four full reference mismatch: {err}")
            refs=range(4) if origin=="AF" else (0,1,3)
            parity=[base._trace_compare(logits=logits[j],trace=trace,batch_index=j,
                        absolute_offset=22,token_id=raw[j]["token_ids"][22],
                        role="original_AF_x1" if origin=="AF" and j==TARGET else "source_companion_x1",
                        atol=TOL) for j in refs]
            require(all(x["passed"] for x in parity),f"{name} source trace parity failed")
            receipt["cells"][-1].update(saved_reference_max_abs=err,source_trace_parity=parity)
            reference[origin]=logits

        caches={};origins={};snapshots={};digests={}
        for name,origin in (("prefill_AF","AF"),("prefill_FF","FF")):
            cache=DynamicCache();inputs=split_inputs(fulls[origin],cache,"prefill")
            invoke(name,inputs,media=True,cache=cache)
            require(cache.get_seq_length()==WIDTH,"prefill cache length changed")
            caches[origin]=cache; origins[origin]=segments(cache)
            snapshots[origin]=blocks(cache,origin)
            digests[origin]=cache_digest(cache)
            receipt["cells"][-1]["cache_digest"]=digests[origin]
        require(all(origins["AF"][i][name][span]==origins["FF"][i][name][span]
                    for i in range(LAYERS) for name in ("keys","values")
                    for span in ("prompt","companions")),
                "prefill target pre-object/companion K/V differ")
        torch.save(snapshots,OUT/"historical-blocks.pt")
        write_new(OUT/"prefill-origins.json",{"segments":origins,"digests":digests,
                 "blocks":bind(OUT/"historical-blocks.pt"),
                 "latest_layer0_equal":{name:origins["AF"][0][name]["latest"]==
                                           origins["FF"][0][name]["latest"]
                                        for name in ("keys","values")},
                 "later_latest_different":any(origins["AF"][i][name]["latest"]!=
                                                origins["FF"][i][name]["latest"]
                                                for i in range(1,LAYERS)
                                                for name in ("keys","values"))})
        for name,base_origin,donor_origin,patch in (
            ("anchor_AF","AF","AF",False),("anchor_FF","FF","FF",False),
            ("sham_AF","AF","AF",True),("sham_FF","FF","FF",True),
            ("hybrid_AF_FF","AF","FF",True),("hybrid_FF_AF","FF","AF",True)):
            require(len(receipt["cells"])==NAMES.index(name) and
                    (name in NAMES[:8] or len(receipt["cells"])>=8),
                    "out-of-order or premature hybrid")
            inputs=split_inputs(fulls[base_origin],caches[base_origin],"suffix")
            logits,consumer=invoke(name,inputs,media=False,cache=caches[base_origin],
                                   base_origin=base_origin,donor_blocks=snapshots[donor_origin],
                                   base_segments=origins[base_origin],original_digest=digests[base_origin],
                                   donor_origin=donor_origin if patch else None)
            require(consumer["restored_digest"]==digests[base_origin],
                    f"{name} historical cache failed restoration")
            if name.startswith("anchor"):
                err=float((logits-reference[base_origin]).abs().max())
                require(err<=TOL,f"{name} cached/full all-row mismatch: {err}")
                receipt["cells"][-1]["cached_full_max_abs"]=err
            elif name.startswith("sham"):
                anchor_name="anchor_"+base_origin
                anchor=torch.load(OUT/"cells"/f"{NAMES.index(anchor_name)+1:02d}-{anchor_name}"/
                                  "full-batch-vocabulary.pt",map_location="cpu",weights_only=True)
                err=float((logits-anchor).abs().max())
                require(err<=TOL,f"{name} sham/anchor all-row mismatch: {err}")
                receipt["cells"][-1]["sham_anchor_max_abs"]=err
            else:
                anchor_name="anchor_"+base_origin
                anchor=torch.load(OUT/"cells"/f"{NAMES.index(anchor_name)+1:02d}-{anchor_name}"/
                                  "full-batch-vocabulary.pt",map_location="cpu",weights_only=True)
                err=max(float((logits[j]-anchor[j]).abs().max()) for j in (0,1,3))
                anchor_consumer=json.loads((OUT/"cells"/
                    f"{NAMES.index(anchor_name)+1:02d}-{anchor_name}"/"consumer.json").read_text())
                require(err<=TOL and [x["companion_suffix"] for x in consumer["after_suffix"]]==
                        [x["companion_suffix"] for x in anchor_consumer["after_suffix"]],
                        f"{name} companion suffix K/V or logits changed")
                receipt["cells"][-1]["companion_anchor_max_abs"]=err
        require(counts=={"model_forwards":10,"vision_forwards":4,"generated_tokens":0},
                "finite model/vision counts changed")
        receipt["effective_identity"]=identity
        receipt["status"]="candidate_raw_complete"
    except Exception as exc:
        receipt["status"]="technical_failure"
        receipt["failure"]={"type":type(exc).__name__,"message":str(exc),
                            "traceback":traceback.format_exc()}
    finally:
        for handle in handles:handle.remove()
        receipt["internal_seconds"]=time.monotonic()-started
        receipt["rss_peak_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch.cuda.is_available():
            receipt["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
            receipt["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device)
        receipt["artifact_bytes_before_receipt"]=sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"]=os.getpid()
        write_new(OUT/"receipt.json",receipt)
        print(json.dumps({"status":receipt["status"],"counts":counts,
                          "failure":receipt.get("failure",{}).get("message")}))
    require(receipt["status"]=="candidate_raw_complete","technical failure; no retry")


def run_parent():
    checked();require(not (OUT/"launch.json").exists() and not (OUT/"outer.json").exists(),
                      "attempt already launched; no retry")
    OUT.mkdir(parents=True,exist_ok=True)
    command=["python","-B","-m","probes.training_set_completion.recurrence_contextual_kv_carrier.run","gpu"]
    began=time.monotonic();unix=time.time()
    with (OUT/"stdout.log").open("x") as stdout,(OUT/"stderr.log").open("x") as stderr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        code=child.wait()
    elapsed=time.monotonic()-began
    outer={"command":command,"started_unix":unix,"outer_seconds":elapsed,
           "child_pid":child.pid,"returncode":code,"terminal":True,
           "stdout":bind(OUT/"stdout.log"),"stderr":bind(OUT/"stderr.log")}
    write_new(OUT/"outer.json",outer)
    print(json.dumps(outer))
    require(code==0,"terminal child failure; no retry")


def reduce_vectors(vectors):
    probs={name:torch.softmax(value[TARGET].double(),dim=-1) for name,value in vectors.items()}
    def tv(x,y):return float((probs[x]-probs[y]).abs().sum()/2)
    baseline=tv("anchor_AF","anchor_FF")
    pairs={"hybrid_AF_FF":("anchor_FF","anchor_AF"),
           "hybrid_FF_AF":("anchor_AF","anchor_FF")}
    outcomes={}
    guard=1e-6
    for name,(latest,older) in pairs.items():
        dl=tv(name,latest);de=tv(name,older)
        rl=dl/baseline if baseline>0 else None
        re=de/baseline if baseline>0 else None
        if baseline<=1e-6:category="negligible_anchor_separation"
        elif min(abs(rl-.5),abs(re-.5),abs((re-rl)-.1),abs((rl-re)-.1),
                 abs(baseline-1e-6))<=guard:category="numerical_HOLD"
        elif rl<.5 and re-rl>.1:category="latest_origin"
        elif re<.5 and rl-re>.1:category="older_origin"
        else:category="neither"
        outcomes[name]={"latest_anchor":latest,"older_anchor":older,
                        "TV_latest":dl,"TV_older":de,"r_L":rl,"r_E":re,
                        "category":category}
    categories=[v["category"] for v in outcomes.values()]
    if baseline<=1e-6:shared="negligible_anchor_separation"
    elif "numerical_HOLD" in categories:shared="numerical_HOLD"
    elif categories==["latest_origin"]*2:shared="primary_latest_origin_pass"
    elif categories==["older_origin"]*2:shared="older_origin_comparator_pass"
    else:shared="mixed_or_neither"
    secondary={}
    for name,value in vectors.items():
        z=value[TARGET].double();top=torch.topk(z,2)
        p=probs[name]
        secondary[name]={"winner":int(top.indices[0]),"runner":int(top.indices[1]),
                         "winner_runner_gap":float(top.values[0]-top.values[1]),
                         "z_151671_minus_151670":float(z[151671]-z[151670]),
                         "fixed_tokens":{str(j):{"probability":float(p[j]),
                                                  "rank":int((z>z[j]).sum())+1}
                                         for j in (151670,151671)}}
    return {"baseline_TV":baseline,"hybrids":outcomes,"shared_outcome":shared,
            "secondary":secondary}


def readback():
    a,p=checked()
    require(not (OUT/"readback.json").exists() and not (UNIT/"candidate-results.md").exists(),
            "cold candidate already exists")
    r=json.loads((OUT/"receipt.json").read_text())
    outer=json.loads((OUT/"outer.json").read_text())
    require(r["status"]=="candidate_raw_complete" and r["admission"]==bind(ADMISSION) and
            r["preflight"]==bind(PREFLIGHT) and r["producer"]==bind(Path(__file__)) and
            r["counts"]=={"model_forwards":10,"vision_forwards":4,"generated_tokens":0} and
            outer["returncode"]==0 and outer["terminal"] and
            outer["child_pid"]==r["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists(),
            "terminal source/receipt/count/cost changed")
    record_guard(r["cells"])
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"cold reader loaded language model")
    batch,raw,trace,sr,_=prior.source(q,a,torch.device("cpu"))
    require(sr["input_identity"]==p["input_identity"] and
            int(q.tokenizer.pad_token_id)==p["pad_id"],"cold source changed")
    fulls={o:make_full(base.ConfigOnlyRope(),batch,raw,p["pad_id"],o) for o in ("AF","FF")}
    pre=json.loads((OUT/"prefill-origins.json").read_text())
    require(pre["blocks"]==bind(OUT/"historical-blocks.pt") and
            set(pre["segments"])==set(pre["digests"])=={"AF","FF"},
            "cold prefill blocks/origins changed")
    blocks_saved=torch.load(pre["blocks"]["path"],map_location="cpu",weights_only=True)
    require(set(blocks_saved)=={"AF","FF"},"cold donor origins changed")
    for origin in ("AF","FF"):
        require(blocks_saved[origin]["origin"]==origin and
                len(blocks_saved[origin]["layers"])==LAYERS and
                len(pre["segments"][origin])==len(pre["digests"][origin])==LAYERS,
                "cold 28-layer donor/cache capture changed")
        for i,row in enumerate(blocks_saved[origin]["layers"]):
            for span in ("older","latest"):
                for name in ("keys","values"):
                    block=row[span][name]
                    require(block.shape==(HEADS,9,DIM) and torch.isfinite(block).all().item() and
                            base.tensor_hash(block)==pre["segments"][origin][i][name][span],
                            "cold donor tensor hash/shape changed")
    require(all(pre["segments"]["AF"][i][name][span]==
                pre["segments"]["FF"][i][name][span]
                for i in range(LAYERS) for name in ("keys","values")
                for span in ("prompt","companions")),
            "cold invariant target prompt/companions failed")
    vectors={};consumer={};records=[]
    for i,cell in enumerate(r["cells"]):
        name=cell["name"];base_origin,donor_origin=ORIGINS[i]
        for key in ("input","vector","consumer"):
            require(bind(cell[key]["path"])==cell[key],f"cold {name} {key} binding changed")
        saved=json.loads(Path(cell["input"]["path"]).read_text())
        recorded=json.loads(Path(cell["consumer"]["path"]).read_text())
        vector=torch.load(cell["vector"]["path"],map_location="cpu",weights_only=True)
        require(recorded["name"]==name and vector.shape==(4,152670) and
                torch.isfinite(vector).all().item() and
                len(recorded["attention"])==LAYERS and
                [x["layer"] for x in recorded["attention"]]==list(range(LAYERS)) and
                len(recorded["after_suffix"])==(LAYERS if i>=4 else 0),
                "cold vector/consumer/layer receipt changed")
        full=fulls[base_origin]
        expected=full if i<2 else split_inputs(full,object(),"prefill" if i<4 else "suffix")
        for key in ("input_ids","attention_mask","position_ids","cache_position"):
            require(saved[key]==expected[key].cpu().tolist() and
                    recorded["actual_input"][key]==base.tensor_hash(expected[key]),
                    f"cold {name} actual source/{key} changed")
        mask=expected_suffix_mask(full) if i>=4 else base.native_4d(expected["attention_mask"])
        require(recorded["expected_mask_sha256"]==base.tensor_hash(mask) and
                all(row["mask_sha256"]==base.tensor_hash(mask) for row in recorded["attention"]),
                f"cold {name} actual all-layer mask changed")
        if i>=4:
            require(recorded["restored_digest"]==pre["digests"][base_origin] and
                    len(recorded["patched_digest"])==LAYERS,
                    f"cold {name} restoration changed")
            for layer,row in enumerate(recorded["attention"]):
                for axis in ("keys","values"):
                    got=row["segments"][axis]
                    want_base=pre["segments"][base_origin][layer][axis]
                    require(all(got[span]==want_base[span] for span in
                                ("prompt","older","companions")) and
                            got["latest"]==base.tensor_hash(
                                blocks_saved[donor_origin]["layers"][layer]["latest"][axis]) and
                            recorded["patched_digest"][layer][axis]==
                                recorded["after_suffix"][layer]["historical_digest"][axis],
                            f"cold {name} layer {layer} donor/complement/after-suffix changed")
        else:
            require(recorded["patched_digest"] is None and
                    recorded["restored_digest"] is None,
                    "cold full/prefill cache receipt changed")
        vectors[name]=vector;consumer[name]=recorded
        records.append({"name":name,"call":i+1,"origins":list(ORIGINS[i]),
                        "vector":cell["vector"],"input":cell["input"],
                        "consumer":cell["consumer"]})
    for name,origin in (("full_AF","AF"),("full_FF","FF")):
        ref=torch.load(a["saved_full_vector_references"]["native_AF" if origin=="AF" else "FF"]["path"],
                       map_location="cpu",weights_only=True)["logits"]
        require(float((vectors[name]-ref).abs().max())<=TOL,"cold saved full reference changed")
        refs=range(4) if origin=="AF" else (0,1,3)
        require(all(base._trace_compare(logits=vectors[name][j],trace=trace,batch_index=j,
                    absolute_offset=22,token_id=raw[j]["token_ids"][22],role="cold_AF_or_companion",
                    atol=TOL)["passed"] for j in refs),"cold original source trace changed")
    for origin in ("AF","FF"):
        require(float((vectors["anchor_"+origin]-vectors["full_"+origin]).abs().max())<=TOL and
                float((vectors["sham_"+origin]-vectors["anchor_"+origin]).abs().max())<=TOL,
                "cold cached/full/sham all-four parity failed")
    for name,origin in (("hybrid_AF_FF","AF"),("hybrid_FF_AF","FF")):
        require(max(float((vectors[name][j]-vectors["anchor_"+origin][j]).abs().max())
                    for j in (0,1,3))<=TOL and
                [x["companion_suffix"] for x in consumer[name]["after_suffix"]]==
                [x["companion_suffix"] for x in consumer["anchor_"+origin]["after_suffix"]],
                "cold hybrid companion state/vector changed")
    reductions=reduce_vectors(vectors)
    result={"status":"candidate_cold_readback_passed","protocol":bind(PROTOCOL),
            "admission":bind(ADMISSION),"preflight":bind(PREFLIGHT),
            "receipt":bind(OUT/"receipt.json"),"outer":bind(OUT/"outer.json"),
            "prefill_origins":bind(OUT/"prefill-origins.json"),
            "cells":records,"counts":r["counts"],"outcomes":reductions,
            "model_loads":0,"model_forwards":0,"vision_forwards":0,
            "cuda_calls":0,"gpu_seconds_added_by_readback":0}
    write_new(OUT/"readback.json",result)
    h=reductions["hybrids"]
    lines=["# Contextual K/V carrier candidate", "",
           "Status: **candidate cold-readback passed; lead acceptance pending.**",
           "",f"Admission SHA `{SHAS[ADMISSION]}`; producer SHA `{bind(Path(__file__))['sha256']}`.",
           f"Raw receipt SHA `{result['receipt']['sha256']}`; readback SHA `{bind(OUT/'readback.json')['sha256']}`.",
           "", "## Frozen endpoint", "",
           f"Anchor TV B = {reductions['baseline_TV']:.12g}; shared outcome **{reductions['shared_outcome']}**.",
           "", "| Hybrid | TV to latest | TV to older | r_L | r_E | Category |",
           "|---|---:|---:|---:|---:|---|" ]
    for name in ("hybrid_AF_FF","hybrid_FF_AF"):
        x=h[name];lines.append(f"| {name} | {x['TV_latest']:.12g} | {x['TV_older']:.12g} | "
                             f"{x['r_L']:.12g} | {x['r_E']:.12g} | {x['category']} |")
    lines += ["", "Full-vocabulary FP64 softmax and TV; both directions retained. "
              "The common header is replayed conditioning, with zero generated tokens. "
              "These x1 distributions do not establish a physical owner or natural mediation percentage.",
              "", "## Qualification and resources", "",
              "All four fresh references/prefills, two cached anchors and two explicit K/V identity shams "
              "qualified before either hybrid; all-four reference/cache/sham max-absolute logit gates use 2e-4. "
              "Actual 28-layer selected K+V, older/prompt/companion complements, native mask/positions and "
              "finally restoration were checked and cold-read separately.",
              "",f"Actual calls {r['counts']['model_forwards']} model / {r['counts']['vision_forwards']} vision / "
              f"{r['counts']['generated_tokens']} generated; parent outer {outer['outer_seconds']:.9f} s; "
              f"internal {r['internal_seconds']:.9f} s. Prior sequence 0.537174243789956 GPUh; "
              f"new cumulative {0.537174243789956+outer['outer_seconds']/3600:.12f} GPUh.",
              f"Peak RSS {r['rss_peak_kib']} KiB; GPU allocated/reserved "
              f"{r['gpu_peak_allocated_bytes']}/{r['gpu_peak_reserved_bytes']} bytes; "
              f"artifact bytes before receipt {r['artifact_bytes_before_receipt']}. "
              f"Terminal child PID {outer['child_pid']}, exit {outer['returncode']}.",
              "", "Individual full vectors, source inputs, all-layer consumers, secondary winner/rank/probability "
              "records and hashes are indexed by the cold [readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-contextual-kv-carrier/attempt-001/readback.json).",
              "", "Original two-record shared conjunction remains NONPASS; F physical owner remains HOLD. "
              "This candidate is not self-accepted and admits no successor.", ""]
    (UNIT/"candidate-results.md").write_text("\n".join(lines))
    print(json.dumps({"status":result["status"],"shared":reductions["shared_outcome"],
                      "baseline_TV":reductions["baseline_TV"],"outer_seconds":outer["outer_seconds"]}))


def selfcheck():
    a=contract()
    q=base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    require(q.model is None,"selfcheck loaded model")
    batch,raw,_,_,_=prior.source(q,a,torch.device("cpu"))
    checks,_=cpu_checks(q,batch,raw,int(q.tokenizer.pad_token_id),a)
    print(json.dumps({"status":"cpu_selfcheck_passed","checks":checks}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("selfcheck","preflight","run","gpu","readback"))
    {"selfcheck":selfcheck,"preflight":preflight,"run":run_parent,
     "gpu":gpu_child,"readback":readback}[p.parse_args().action]()


if __name__=="__main__":main()
