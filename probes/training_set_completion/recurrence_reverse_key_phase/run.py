"""One native U prefill and ten qualified cached S readouts."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import resource
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.recurrence_first_arrivals.prepare import _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest
from probes.training_set_completion.recurrence_key_phase import rotate, unrotate
from probes.training_set_completion.recurrence_next_history_prediction.run import (
    boundary, contract as source_contract, make_input,
)
from probes.training_set_completion.recurrence_position_history import swap_prefix_positions
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-reverse-key-phase"
PROTOCOL = UNIT / "unit.md"
MANIFEST = UNIT / "manifest.json"
CPU_QUALIFICATION = UNIT / "supporting/cpu-qualification-v2.json"
PROTOCOL_SHA = "5df865e7ba8e65a5725831eadcb0191f96070fd472a1ab08bb7136e368504d08"
MANIFEST_SHA = "d746713cb34a726742102ac29c0a8cdfcf981ccfe948cf3b24105735c518bb6c"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-reverse-key-phase/attempt-001")
WIDTH, FULL, S = 1416, 1421, 5
LAYERS, HEADS, DIM = 28, 8, 128
SPANS = {"earlier": (1398,1407), "latest": (1407,1416)}
PHASE = {"earlier": list(range(423,432)), "latest": list(range(432,441)),
         "destination_earlier": list(range(432,441)),
         "destination_latest": list(range(441,450)),
         "suffix_native": list(range(441,446)), "suffix_cross": list(range(432,437))}
TOL, CAP_SECONDS = 2e-4, 900
IMPORTS = [
    "probes/training_set_completion/recurrence_reverse_key_phase/run.py",
    "probes/training_set_completion/recurrence_recent_key_phase/run.py",
    "probes/training_set_completion/recurrence_next_step_cancellation/run.py",
    "probes/training_set_completion/recurrence_next_history_prediction/run.py",
    "probes/training_set_completion/recurrence_chair_history_position/run.py",
    "probes/training_set_completion/recurrence_key_phase.py",
    "probes/training_set_completion/recurrence_history_cache_partition.py",
    "probes/training_set_completion/recurrence_position_history.py",
    "probes/training_set_completion/artifacts.py",
    "probes/training_set_completion/coordinate_continuity/runtime.py",
    "probes/training_set_completion/native_row_choice/runtime.py",
    "probes/training_set_completion/numerical_feedback/select.py",
    "probes/training_set_completion/recurrence_first_arrivals/prepare.py",
    "probes/training_set_completion/recurrence_first_arrivals/stage1_case.py",
    "probes/training_set_completion/untied_shared.py",
    "src/artifacts/source_provenance.py", "src/qwen/input_identity.py",
    "src/qwen/native.py", "src/qwen/runtime_loading.py", "src/inference/bound_requests.py",
]


def bound(item):
    p=Path(item["path"])
    if not p.is_absolute():p=REPO/p
    got=literal_binding(p)
    _require(got["sha256"]==item["sha256"] and
             ("size_bytes" not in item or got["size_bytes"]==item["size_bytes"]),
             f"binding changed: {p}")
    return p


def contract():
    _require(literal_binding(PROTOCOL)["sha256"]==PROTOCOL_SHA and
             literal_binding(MANIFEST)["sha256"]==MANIFEST_SHA,"lead protocol/manifest changed")
    m=json.loads(MANIFEST.read_text())
    _require(m["status"]=="lead-frozen-before-reverse-treatment" and
             m["source_group"]=="refined-04" and m["batch_index"]==0 and
             m["request_id"]=="coco2017_train_000000477415" and
             [(p["owner"],p["x1_token"]) for p in m["probes"]]==[(1589003,152180),(1586761,152088)] and
             m["geometry"]=={"prompt_width":1362,"history_width":1416,"suffix_width":5,
                 "earlier_raw":[36,45],"latest_raw":[45,54],
                 "earlier_physical":[1398,1407],"latest_physical":[1407,1416],
                 "earlier_phase":PHASE["earlier"],"latest_phase":PHASE["latest"],
                 "suffix_native_phase":PHASE["suffix_native"],
                 "suffix_crossed_phase":PHASE["suffix_cross"],"phase_shift":9,
                 "earlier_destination_phase":PHASE["destination_earlier"],
                 "latest_destination_phase":PHASE["destination_latest"],
                 "treatment_suffix_phase":PHASE["suffix_native"]} and
             m["decision"]=={"target":"qualified crossed U/L full-vocabulary distribution","baseline":"TV(native,cross)",
                 "maximum_latest_error_over_baseline":.5,
                 "minimum_earlier_minus_latest_error_over_baseline":.1,
                 "numerical_boundary_guard":1e-6,
                 "shared_requires":"both probes"} and
             m["budget"]["prefill_model_forwards"]==1 and
             m["budget"]["suffix_model_forwards"]==10 and
             m["budget"]["vision_forwards"]==1 and m["budget"]["free_tokens"]==0 and
             m["budget"]["incremental_gpu_hours_cap"]==.25 and
             m["budget"]["sequence_gpu_hours_cap"]==8,
             "finite geometry/decision/budget changed")
    for k in ("predecessor_acceptance","predecessor_manifest","predecessor_reduction"):
        bound(m[k])
    for k in ("raw","trace","runtime_receipt","image"):
        bound(m["source_bindings"][k])
    old,receipt,native,panel=source_contract()
    _require(m["source_bindings"]==old["source_bindings"] and
             native[36:45]==native[45:54] and len(native[36:45])==9 and
             receipt["input_identity"]["request_ids"]==[m["request_id"]],
             "single-request A/U source changed")
    for p in m["probes"]:
        refs=m["references"][str(p["owner"])]
        _require(set(refs)=={"native","cross"},"anchor set changed")
        for name in ("native","cross"):
            for k in ("vector","consumer"):bound(refs[name][k])
    return m,receipt,native,panel


def plan(m):
    probes=[p["owner"] for p in m["probes"]]
    return ([(o,"native") for o in probes]+[(o,"cross") for o in probes]+
            [(o,"sham") for o in probes]+
            [(o,"latest") for o in probes]+[(o,"earlier") for o in probes])


def checked_patch_span(name,span,phase_positions):
    _require(name in SPANS and span==SPANS[name] and
             phase_positions==PHASE[f"destination_{name}"] and
             span[1]-span[0]==9 and span[1]<=WIDTH,
             "wrong row, nine-key span or plus9 phase")


def checked_suffix(kind,positions):
    _require(kind in ("native","cross","sham","latest","earlier") and
             positions==(PHASE["suffix_cross"] if kind=="cross" else PHASE["suffix_native"]),
             "wrong native/cross S position")


def cpu_mutations():
    for name,span,phase in (("latest",SPANS["earlier"],PHASE["destination_latest"]),
                             ("latest",SPANS["latest"],PHASE["latest"]),
                             ("earlier",SPANS["earlier"],PHASE["earlier"]),
                             ("latest",SPANS["latest"],PHASE["suffix_native"])):
        try:checked_patch_span(name,span,phase)
        except (AssertionError,ValueError):pass
        else:raise AssertionError("wrong row or phase escaped CPU check")
    for name in SPANS:checked_patch_span(name,SPANS[name],PHASE[f"destination_{name}"])
    for kind in ("native","cross","sham","latest","earlier"):
        checked_suffix(kind,PHASE["suffix_cross"] if kind=="cross" else PHASE["suffix_native"])
    for kind,phase in (("latest",PHASE["suffix_cross"]),("sham",PHASE["suffix_cross"]),
                       ("cross",PHASE["suffix_native"])):
        try:checked_suffix(kind,phase)
        except (AssertionError,ValueError):pass
        else:raise AssertionError("wrong S position escaped CPU check")
    return ["wrong_A_span_rejected","wrong_sign_rejected","clipped_phase_rejected",
            "wrong_native_S_rejected"]


def destination_phase(rotary,device):
    """Use the loaded rotary module; independently reconstruct all nine positions in FP64."""
    positions=PHASE["destination_earlier"]+PHASE["destination_latest"]
    ids=torch.tensor([positions]*3,device=device,dtype=torch.long).unsqueeze(1)
    with torch.no_grad():
        cos,sin=rotary(torch.empty((1,len(positions),1),device=device,dtype=torch.float32),ids)
    _require(cos.shape==sin.shape==(1,18,DIM) and cos.dtype==sin.dtype==torch.float32,
             "destination rotary output shape/dtype changed")
    frequencies=rotary.inv_freq.detach().double().cpu()
    sections=list(rotary.mrope_section)
    _require(frequencies.shape==(DIM//2,) and len(sections)==3 and
             sum(sections)==DIM//2,"actual model frequency/layout changed")
    base=torch.tensor(positions,dtype=torch.float64)[:,None]*frequencies[None,:]
    axes=base.repeat(3,1,1)
    interleaved=axes[0].clone()
    for axis in (1,2):
        idx=list(range(axis,3*sections[axis],3))
        interleaved[:,idx]=axes[axis,:,idx]
    angle=torch.cat((interleaved,interleaved),dim=-1)
    scale=float(rotary.attention_scaling)
    error=max(float((cos[0].double().cpu()-angle.cos()*scale).abs().max()),
              float((sin[0].double().cpu()-angle.sin()*scale).abs().max()))
    _require(error<=TOL,"independent FP64 full destination phase mismatch")
    values={name:{"cos":cos[0,i*9:(i+1)*9].detach().float().cpu().clone(),
                  "sin":sin[0,i*9:(i+1)*9].detach().float().cpu().clone()}
            for i,name in enumerate(("destination_earlier","destination_latest"))}
    return values,{"positions":[positions]*3,"dtype":"float32",
                   "inv_freq_sha256":tensor_hash(rotary.inv_freq),
                   "mrope_section":sections,"attention_scaling":scale,
                   "fp64_full_destination_max_abs_error":error,"rotary_module_invocations":1}


def preflight(out):
    m,receipt,native,panel=contract()
    _require(not out.exists(),"attempt already exists")
    checks=cpu_mutations()
    cpu=json.loads(CPU_QUALIFICATION.read_text())
    _require(cpu["status"]=="passed" and cpu["producer_sha256"]==literal_binding(Path(__file__))["sha256"] and
             cpu["mutation_checks"]==checks and cpu["model_forwards"]==cpu["vision_forwards"]==
             cpu["cuda_calls"]==0 and cpu["patch_normal_restored"] and
             cpu["patch_forced_exception_restored"] and cpu["unchanged_V_and_unselected_K"] and
             cpu["destination_full_positions"]==PHASE["destination_latest"],
             "CPU route qualification changed")
    q=load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    _require(q.model is None,"CPU preflight loaded model")
    batch,raw,trace,group,planning=_source(boundary(m,native),"untied",panel,q,torch.device("cpu"))
    _require(len(raw)==len(group["cases"])==1 and
             input_identity(batch)==receipt["input_identity"] and
             len(batch.prompt_token_ids[0])==1362,"single request/image changed")
    prompt=list(batch.prompt_token_ids[0]);crosswalk=[]
    previous=json.loads(bound(m["predecessor_manifest"]).read_text())
    for p in m["probes"]:
        owner=p["owner"]
        for ref_name in ("native","cross"):
            ref=m["references"][str(owner)][ref_name]
            saved=json.loads(bound(ref["consumer"]).read_text())
            full_name="U/U" if ref_name=="native" else "U/L"
            full=json.loads(bound(previous["references"][str(owner)][full_name]["actual_consumer"]).read_text())
            expected=prompt+native[:54]+native[54:58]+[p["x1_token"]]
            pos=PHASE["suffix_native"] if ref_name=="native" else PHASE["suffix_cross"]
            _require(full["input_ids"]==[expected] and
                     saved["input_ids"]==[expected[-S:]] and
                     full["attention_mask"]==saved["attention_mask"]==[[1]*FULL] and
                     saved["position_ids"]==[[pos]]*3 and
                     [axis[0][-S:] for axis in full["position_ids"]]==[pos]*3 and
                     [axis[0][1398:1407] for axis in full["position_ids"]]==[PHASE["earlier"]]*3 and
                     [axis[0][1407:1416] for axis in full["position_ids"]]==[PHASE["latest"]]*3,
                     "accepted U reference consumer changed")
            crosswalk.append({"owner":owner,"cell":ref_name,"consumer":ref["consumer"],
                              "full_consumer":previous["references"][str(owner)][full_name]["actual_consumer"],
                              "vector":ref["vector"],"S_physical":[1416,1421],
                              "S_rotary_positions":[pos]*3})
    out.mkdir(parents=True)
    sources=[REPO/x for x in IMPORTS]
    sources += [Path(inspect.getfile(x)) for x in (cache_utils,modeling_qwen3_vl,sdpa_attention)]
    captures=[]
    for source in sources:
        relative=source.relative_to(REPO) if source.is_relative_to(REPO) else Path("transformers")/source.name
        capture=preserve_source(source,run_root=out,relative_name=relative)
        captures.append({"maintained":literal_binding(source),"capture":literal_binding(capture)})
    old=json.loads(bound(m["predecessor_reduction"]).read_text())
    prior=json.loads(bound(old["pilot"]).read_text())
    _require(prior["status"]=="candidate_complete" and
             prior["cost"]["model_forwards"]==11 and prior["cost"]["vision_forwards"]==1,
             "matched predecessor cost source changed")
    estimate=2*prior["cost"]["allocated_gpu_seconds"]
    _require(estimate<CAP_SECONDS and
             m["budget"]["sequence_cumulative_prior_gpu_hours"]+estimate/3600<8,
             "shape-aware cost forecast exceeds budget")
    packet={"schema":"recurrence_reverse_key_phase.preflight.v1","status":"cpu_qualified_before_gpu",
            "protocol":literal_binding(PROTOCOL),"manifest":literal_binding(MANIFEST),
            "source":m["source_bindings"],"input_identity":input_identity(batch),
            "source_planning":planning,"shape":{"batch_size":1,"prompt_width":1362,
                "history_width":WIDTH,"full_width":FULL,"image_grids":[list(x) for x in batch.image_grids],
                "pixel_elements":int(batch.inputs["pixel_values"].numel())},
            "crosswalk":crosswalk,"cpu_qualification":literal_binding(CPU_QUALIFICATION),
            "cpu_mutation_checks":checks,"direct_source_captures":captures,
            "effective_identity_expected":receipt["identity"],"decision":m["decision"],
            "phase_generation":{"route":"loaded Qwen3VLTextRotaryEmbedding forward with explicit 3x1x18 positions",
                "destination_earlier":PHASE["destination_earlier"],
                "destination_latest":PHASE["destination_latest"],
                "independent_oracle":"FP64 actual inv_freq and mrope_section, all 18 positions",
                "max_abs_error":TOL,"model_forwards":0,"vision_forwards":0},
            "cost_forecast":{"matched_predecessor_pilot":old["pilot"],
                "old_batch_size":1,"old_full_width":FULL,"old_forwards":11,"old_vision":1,
                "old_allocated_seconds":prior["cost"]["allocated_gpu_seconds"],"new_prefill":1,"new_suffix":10,
                "size_credit":0,"planning_multiplier":2,"estimated_seconds":estimate,
                "incremental_cap_seconds_including_failure":CAP_SECONDS,
                "cumulative_prior_gpu_hours":m["budget"]["sequence_cumulative_prior_gpu_hours"],
                "artifact_planning_bytes":m["budget"]["artifact_planning_bytes"]},
            "commands":{"gpu":["python","-B","-m","probes.training_set_completion.recurrence_reverse_key_phase.run","run","--output",str(out),"--device","cuda:0"],
                        "readback":["python","-B","-m","probes.training_set_completion.recurrence_reverse_key_phase.run","readback","--output",str(out)],
                        "reduce":["python","-B","-m","probes.training_set_completion.recurrence_reverse_key_phase.run","reduce","--output",str(out)]}}
    _write_new(out/"preflight.json",packet)
    print(json.dumps({"status":packet["status"],"source_shape":packet["shape"],
                      "estimated_seconds":estimate,"captures":len(captures)}))


def complex_rephase(post,old_cos,old_sin,new_cos,new_sin):
    post=post.double();a,b=post[...,:DIM//2],post[...,DIM//2:]
    c,s=old_cos.double()[...,:DIM//2],old_sin.double()[...,:DIM//2]
    denominator=c.square()+s.square()
    _require(bool(torch.all(denominator>0)),"invalid observed phase denominator")
    real=(a*c.unsqueeze(0)+b*s.unsqueeze(0))/denominator.unsqueeze(0)
    imag=(b*c.unsqueeze(0)-a*s.unsqueeze(0))/denominator.unsqueeze(0)
    c,s=new_cos.double()[...,:DIM//2],new_sin.double()[...,:DIM//2]
    return torch.cat((real*c.unsqueeze(0)-imag*s.unsqueeze(0),
                      imag*c.unsqueeze(0)+real*s.unsqueeze(0)),dim=-1)


def qualify_phase(blocks):
    eps=torch.finfo(torch.float32).eps
    records=[];candidates={"latest":[],"earlier":[],"sham":[]}
    phase=blocks["phase"]
    for i,item in enumerate(blocks["layers"]):
        row_record={"layer":i}
        for name in ("earlier","latest"):
            own=phase[name];donor=phase[f"destination_{name}"]
            pre=item[name]["pre_k"];post=item[name]["native_k"]
            native=rotate(pre,own["cos"],own["sin"])
            recovered=unrotate(post,own["cos"],own["sin"])
            identity=rotate(recovered,own["cos"],own["sin"])
            changed=rotate(pre,donor["cos"],donor["sin"])
            inverse=rotate(unrotate(changed,donor["cos"],donor["sin"]),own["cos"],own["sin"])
            oracle=complex_rephase(post,own["cos"],own["sin"],donor["cos"],donor["sin"])
            key_scale=max(1.,float(pre.abs().max()),float(post.abs().max()))
            post_scale=max(1.,float(post.abs().max()))
            norm_scale=max(1.,float(post.norm(dim=-1).max()))
            measured={"native_replay":float((native-post).abs().max()),
                      "inverse_preK":float((recovered-pre).abs().max()),
                      "roundtrip":float((inverse-post).abs().max()),
                      "norm":float((changed.norm(dim=-1)-post.norm(dim=-1)).abs().max()),
                      "fp64_crossed":float((changed.double()-oracle).abs().max())}
            bounds={"native_replay":2e-5,"inverse_preK":8*eps*key_scale,
                    "roundtrip":8*eps*key_scale,"norm":8*eps*norm_scale,
                    "fp64_crossed":8*eps*post_scale}
            row_record[name]={"measured":measured,"bounds":bounds,
                              "ratios":{k:measured[k]/bounds[k] for k in bounds},
                              "qualified":all(measured[k]<=bounds[k] for k in bounds)}
            candidates[name].append(changed.detach().clone())
            if name=="latest":candidates["sham"].append(identity.detach().clone())
        records.append(row_record)
    return records,candidates


@contextmanager
def patched(cache,name,keys,native_digest):
    """Patch and restore inside one inference scope, including the caller body."""
    with torch.inference_mode():
        with _patched(cache,name,keys,native_digest):
            yield


@contextmanager
def _patched(cache,name,keys,native_digest):
    if name in ("native","cross"):
        _require(keys is None,"anchor unexpectedly patched")
        span=None
    else:
        row="latest" if name=="sham" else name
        span=SPANS[row]
        if name!="sham":checked_patch_span(row,span,PHASE[f"destination_{row}"])
        _require(len(keys)==LAYERS,"candidate layer count changed")
    saved=[]
    try:
        if span is not None:
            for layer,key in zip(cache.layers,keys,strict=True):
                _require(key.shape==(HEADS,9,DIM),"candidate K block shape changed")
                old=layer.keys[0,:,span[0]:span[1],:].clone()
                saved.append((layer,old))
                layer.keys[0,:,span[0]:span[1],:].copy_(key)
        yield
    finally:
        cache.crop(WIDTH)
        if span is not None:
            for layer,old in saved:
                layer.keys[0,:,span[0]:span[1],:].copy_(old)
        _require(cache.get_seq_length()==WIDTH and cache_digest(cache)==native_digest,
                 "native K/V cache did not restore after suffix")


def run(out,device_name):
    started=time.monotonic()
    m,source_receipt,native,panel=contract()
    pre_path=out/"preflight.json";pre=json.loads(pre_path.read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and pre["manifest"]==literal_binding(MANIFEST) and
             pre["decision"]==m["decision"] and device_name=="cuda:0" and
             not (out/"receipt.json").exists(),"GPU authority changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]);bound(item["capture"])
    launch=_write_new(out/"launch.json",{"status":"frozen_before_model_load","pid":os.getpid(),
                       "started_unix":time.time(),"preflight":literal_binding(pre_path),
                       "producer":literal_binding(Path(__file__)),"hard_cap_seconds":CAP_SECONDS})
    device=torch.device(device_name);counts={"model_forwards":0,"vision_forwards":0};done=[];handles=[];active=None
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        q,identity=load_model("untied",device)
        saved=source_receipt["identity"]
        _require({k:v for k,v in identity.items() if k!="loader_source"}==
                 {k:v for k,v in saved.items() if k!="loader_source"} and
                 all(identity["loader_source"][k]==saved["loader_source"][k]
                     for k in ("sha256","size_bytes")),"effective model identity changed")
        model=q.model.eval()
        batch,raw,trace,group,planning=_source(boundary(m,native),"untied",panel,q,device)
        _require(len(raw)==len(group["cases"])==1 and
                 input_identity(batch)==source_receipt["input_identity"]==pre["input_identity"],
                 "GPU source identity changed")
        pad=int(q.tokenizer.pad_token_id)
        full={}
        previous=json.loads(bound(m["predecessor_manifest"]).read_text())
        for p in m["probes"]:
            owner=p["owner"]
            original=make_input(model,batch,native,6,p["x1_token"],pad)
            donor=torch.tensor([list(range(432,437))]*3,device=device,dtype=original["position_ids"].dtype)
            crossed=swap_prefix_positions(original,donor,0,S,tuple(original["input_ids"][0,-S:].tolist()))
            _require(original["input_ids"].shape==(1,FULL) and
                     original["position_ids"][:,0,-S:].tolist()==[list(range(441,446))]*3 and
                     crossed["position_ids"][:,0,-S:].tolist()==[list(range(432,437))]*3 and
                     torch.equal(crossed["position_ids"][:,:,:-S],original["position_ids"][:,:,:-S]),
                     "U/U or U/L position crosswalk changed")
            full[owner,"native"]=original
            full[owner,"cross"]=crossed
            for name,inp in (("native",original),("cross",crossed)):
                full_name="U/U" if name=="native" else "U/L"
                saved_consumer=json.loads(bound(previous["references"][str(owner)][full_name]["actual_consumer"]).read_text())
                _require(inp["input_ids"].cpu().tolist()==saved_consumer["input_ids"] and
                         inp["position_ids"].cpu().tolist()==saved_consumer["position_ids"] and
                         inp["attention_mask"].cpu().tolist()==saved_consumer["attention_mask"],
                         "full accepted anchor input changed")
        cache=DynamicCache()
        base=full[1589003,"native"]
        prefill=dict(base)
        prefill.update(input_ids=base["input_ids"][:,:WIDTH],
                       attention_mask=base["attention_mask"][:,:WIDTH],
                       position_ids=base["position_ids"][:,:,:WIDTH],
                       cache_position=torch.arange(WIDTH,device=device),
                       past_key_values=cache,use_cache=True,logits_to_keep=1)
        _require(prefill["input_ids"].shape==(1,WIDTH) and
                 prefill["input_ids"][0].tolist()==base["input_ids"][0,:WIDTH].tolist(),
                 "prefill history changed")
        def before_model(_module,_args,kwargs):
            counts["model_forwards"]+=1
            _require(active is not None and counts["model_forwards"]<=11 and
                     time.monotonic()-started<CAP_SECONDS,"unexpected or over-cap forward")
            for key in ("input_ids","attention_mask","position_ids","cache_position"):
                _require(torch.equal(kwargs.get(key),active["input"][key]),
                         f"actual model consumer {key} changed")
            _require(kwargs.get("past_key_values") is cache and kwargs.get("use_cache") is True,
                     "actual model consumed another cache or disabled it")
            if active["name"]=="prefill":
                for key in ("pixel_values","image_grid_thw"):
                    _require(tensor_hash(kwargs.get(key))==tensor_hash(active["input"][key]),
                             f"prefill media {key} changed")
            else:
                _require("pixel_values" not in kwargs and "image_grid_thw" not in kwargs,
                         "suffix unexpectedly carries vision fields")
            active["observed"]={key:kwargs[key].detach().cpu().tolist()
                                for key in ("input_ids","attention_mask","position_ids","cache_position")}
            active["observed"]["media_present"]=active["name"]=="prefill"
        def before_vision(_module,_args):
            counts["vision_forwards"]+=1
            _require(counts["vision_forwards"]<=1,"vision forward ceiling reached")
        handles.extend((model.register_forward_pre_hook(before_model,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision)))
        phase_seen=[];pre_norm=[{} for _ in range(LAYERS)];prefill_hooks=[]
        def prefill_rotary(_module,_args,output):
            cos,sin=output
            _require(cos.shape==sin.shape==(1,WIDTH,DIM),"prefill rotary shape changed")
            for name,span in SPANS.items():
                values=base["position_ids"][:,0,span[0]:span[1]].tolist()
                _require(values==[PHASE[name]]*3,"prefill observed phase donor position changed")
            phase_seen.append({name:{"cos":cos[0,span[0]:span[1]].detach().clone(),
                                     "sin":sin[0,span[0]:span[1]].detach().clone()}
                               for name,span in SPANS.items()})
        prefill_hooks.append(model.model.language_model.rotary_emb.register_forward_hook(prefill_rotary))
        for i,layer in enumerate(model.model.language_model.layers):
            def pre_key(_module,_args,output,index=i):
                _require(output.shape==(1,WIDTH,HEADS,DIM),"prefill normalized pre-K shape changed")
                pre_norm[index]={name:output[0,span[0]:span[1]].transpose(0,1).detach().clone()
                                 for name,span in SPANS.items()}
            prefill_hooks.append(layer.self_attn.k_norm.register_forward_hook(pre_key))
        active={"name":"prefill","input":prefill,"observed":{}}
        try:
            with torch.inference_mode():output=model(**prefill)
        finally:
            for hook in prefill_hooks:hook.remove()
        torch.cuda.synchronize(device)
        _require(output.past_key_values is cache and len(phase_seen)==1 and
                 all(set(x)==set(SPANS) for x in pre_norm) and
                 counts=={"model_forwards":1,"vision_forwards":1},"prefill capture incomplete")
        _require(len(cache.layers)==LAYERS and all(
            isinstance(getattr(layer,axis),torch.Tensor) and
            getattr(layer,axis).shape==(1,HEADS,WIDTH,DIM) and
            getattr(layer,axis).dtype==torch.float32
            for layer in cache.layers for axis in ("keys","values")),
            "single-request prefill K/V shape or dtype changed")
        native_digest=cache_digest(cache)
        native_segments=[{"before":tensor_hash(layer.keys[:,:,:SPANS["earlier"][0],:]),
                          **{name:tensor_hash(layer.keys[:,:,a:b,:]) for name,(a,b) in SPANS.items()},
                          "values":tensor_hash(layer.values)} for layer in cache.layers]
        destinations,destination_record=destination_phase(model.model.language_model.rotary_emb,device)
        earlier_error=max(float((destinations["destination_earlier"][axis]-
                                 phase_seen[0]["latest"][axis].cpu()).abs().max())
                          for axis in ("cos","sin"))
        _require(earlier_error<=TOL,"earlier destination differs from observed latest phase")
        destination_record["earlier_matches_observed_latest_max_abs_error"]=earlier_error
        torch.save(destinations,out/"destination-phase.pt")
        _write_new(out/"destination-phase-raw.json",{
            **destination_record,"vector":literal_binding(out/"destination-phase.pt")})
        phase_seen[0].update(destinations)
        blocks={"phase":{name:{key:value.detach().float().cpu() for key,value in rec.items()}
                         for name,rec in phase_seen[0].items()},"layers":[]}
        for i,layer in enumerate(cache.layers):
            blocks["layers"].append({name:{"pre_k":pre_norm[i][name].float().cpu(),
                                           "native_k":layer.keys[0,:,span[0]:span[1],:].detach().float().cpu().clone(),
                                           "native_v":layer.values[0,:,span[0]:span[1],:].detach().float().cpu().clone()}
                                     for name,span in SPANS.items()})
        torch.save(blocks,out/"prefill-blocks.pt")
        _write_new(out/"prefill-raw.json",{"vector":literal_binding(out/"prefill-blocks.pt"),
                   "consumer":active["observed"],"cache_digest":native_digest,
                   "native_segments":native_segments,
                   "counts_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started})
        active=None
        phase_records,candidate=qualify_phase(blocks)
        _write_new(out/"phase-qualification.json",{"records":phase_records,
                   "prefill_blocks":literal_binding(out/"prefill-blocks.pt")})
        _require(all(v["qualified"] for rec in phase_records for k,v in rec.items() if k!="layer"),
                 "scale-aware or FP64 phase qualification failed")
        completed_anchors={}
        native_s_checks=[]
        for index,(owner,kind) in enumerate(plan(m),1):
            if index>4:
                _require(len(completed_anchors)==4,"cached/full anchors incomplete before patch")
            if index>6:
                _require(len([x for x in done if x["kind"]=="sham"])==2,
                         "both identity shams incomplete before treatment")
            actual_kind="cross" if kind=="cross" else "native"
            parent=full[owner,actual_kind]
            checked_suffix(kind,parent["position_ids"][0,0,WIDTH:FULL].tolist())
            suffix={"input_ids":parent["input_ids"][:,WIDTH:FULL],
                    "attention_mask":parent["attention_mask"],
                    "position_ids":parent["position_ids"][:,:,WIDTH:FULL],
                    "cache_position":torch.arange(WIDTH,FULL,device=device),
                    "past_key_values":cache,"use_cache":True,"return_dict":True,"logits_to_keep":1}
            edited=None if kind in ("native","cross") else candidate["sham" if kind=="sham" else kind]
            with patched(cache,kind,edited,native_digest):
                expected=cache_digest(cache)
                for layer_index,layer in enumerate(cache.layers):
                    unchanged=("earlier",) if kind in ("sham","latest") else (
                        ("latest",) if kind=="earlier" else ("earlier","latest"))
                    _require(tensor_hash(layer.keys[:,:,:SPANS["earlier"][0],:])==
                             native_segments[layer_index]["before"] and
                             all(tensor_hash(layer.keys[:,:,SPANS[name][0]:SPANS[name][1],:])==
                                 native_segments[layer_index][name] for name in unchanged) and
                             tensor_hash(layer.values)==native_segments[layer_index]["values"],
                             "unselected historical K or V changed before actual consumer")
                active={"name":kind,"input":suffix,"observed":{}}
                before_rows={};after_rows={};rotary=[];rotary_values=[];local=[]
                def rotary_hook(_module,args,output):
                    _require(torch.equal(args[1],suffix["position_ids"]),"actual S rotary position changed")
                    cos,sin=output
                    _require(cos.shape==sin.shape==(1,S,DIM),"actual S rotary shape changed")
                    rotary.append({"cos":tensor_hash(cos),"sin":tensor_hash(sin)})
                    rotary_values.append((cos,sin))
                local.append(model.model.language_model.rotary_emb.register_forward_hook(rotary_hook))
                for layer_index,layer in enumerate(model.model.language_model.layers):
                    def before_attention(_module,_args,kwargs,i=layer_index):
                        _require(kwargs.get("past_key_values") is cache and
                                 cache.get_seq_length(i)==WIDTH and
                                 torch.equal(kwargs.get("cache_position"),suffix["cache_position"]),
                                 "actual attention cache/physical slots changed")
                        mask=kwargs.get("attention_mask")
                        _require(isinstance(mask,torch.Tensor) and mask.dtype==torch.bool and
                                 mask.shape==(1,1,S,FULL),"actual attention mask changed")
                        allowed=torch.arange(FULL,device=mask.device)[None,:] <= (
                            WIDTH+torch.arange(S,device=mask.device)[:,None])
                        _require(torch.equal(mask[0,0],allowed),"actual S causal visibility changed")
                        embeddings=kwargs.get("position_embeddings")
                        _require(len(rotary)==1 and isinstance(embeddings,tuple) and len(embeddings)==2 and
                                 torch.equal(embeddings[0],rotary_values[0][0]) and
                                 torch.equal(embeddings[1],rotary_values[0][1]),
                                 "actual rotary embeddings absent")
                        current=cache.layers[i]
                        observed={axis:tensor_hash(getattr(current,axis)) for axis in ("keys","values")}
                        _require(observed==expected[i],"attention consumed wrong historical K/V")
                        before_rows[i]={"key_sha256":observed["keys"],"value_sha256":observed["values"],
                                        "mask_sha256":tensor_hash(mask),
                                        "cache_position_sha256":tensor_hash(kwargs["cache_position"]),
                                        "selected_K_sha256":{n:tensor_hash(current.keys[0,:,a:b,:])
                                                              for n,(a,b) in SPANS.items()},
                                        "selected_V_sha256":{n:tensor_hash(current.values[0,:,a:b,:])
                                                              for n,(a,b) in SPANS.items()}}
                    def before_output(_module,_args,i=layer_index):
                        _require(cache.get_seq_length(i)==FULL,"S keys not appended before output")
                        current=cache.layers[i]
                        observed={axis:tensor_hash(getattr(current,axis)[:,:,:WIDTH,:])
                                  for axis in ("keys","values")}
                        _require(observed==expected[i],"historical K/V changed during S attention")
                        after_rows[i]={"historical_key_sha256":observed["keys"],
                                       "historical_value_sha256":observed["values"],
                                       "cache_length":cache.get_seq_length(i)}
                    local.extend((layer.self_attn.register_forward_pre_hook(before_attention,with_kwargs=True),
                                  layer.self_attn.o_proj.register_forward_pre_hook(before_output)))
                try:
                    with torch.inference_mode():vec=model(**suffix).logits[0,-1].detach().float().cpu()
                finally:
                    for hook in local:hook.remove()
                torch.cuda.synchronize(device)
                _require(len(rotary)==1 and len(before_rows)==len(after_rows)==LAYERS and
                         counts["vision_forwards"]==1,"actual all-layer S consumer incomplete")
                if kind=="native":
                    native_s_error=max(float((rotary_values[0][j][0].detach().cpu()-
                                             destinations["destination_latest"][axis][:S]).abs().max())
                                       for j,axis in enumerate(("cos","sin")))
                    _require(native_s_error<=TOL,"latest full destination first five differ from actual native S")
                    native_s_checks.append({"owner":owner,"max_abs_error":native_s_error})
                cell_dir=out/"cells"/f"{index:02d}-{owner}-{kind}"
                cell_dir.mkdir(parents=True,exist_ok=False)
                torch.save(vec,cell_dir/"vocabulary.pt")
                vector=literal_binding(cell_dir/"vocabulary.pt")
                consumer=_write_new(cell_dir/"consumer-raw.json",{
                    **active["observed"],"rotary_output_hashes":rotary,
                    "before_attention":before_rows,"after_attention":after_rows,
                    "expected_historical_digest":expected})
                _require(vec.ndim==1 and torch.isfinite(vec).all(),"invalid full vocabulary vector")
                top=torch.topk(vec,2)
                record={"owner":owner,"kind":kind,"vector":vector,"consumer":consumer,
                        "top2_ids":top.indices.tolist(),"top2_logits":top.values.tolist(),
                        "top2_gap":float(top.values[0]-top.values[1]),
                        "logsumexp":float(torch.logsumexp(vec,-1)),"vocabulary_size":int(vec.numel()),
                        "counters_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started}
                if kind in ("native","cross","sham"):
                    ref_name="cross" if kind=="cross" else "native"
                    ref=m["references"][str(owner)][ref_name]["vector"]
                    reference=torch.load(bound(ref),map_location="cpu",weights_only=True)
                    error=float((vec-reference).abs().max())
                    record["full_reference_max_abs_error"]=error
                    _require(error<=TOL,"cached/full or identity-sham vector mismatch")
                    if kind in ("native","cross"):
                        completed_anchors[owner,kind]=True
                _write_new(cell_dir/"cell.json",record)
                done.append({"owner":owner,"kind":kind,"record":literal_binding(cell_dir/"cell.json")})
                _write_new(out/f"checkpoint-{index:02d}.json",{
                    "completed":done,"counts":dict(counts),
                    "allocated_gpu_seconds":time.monotonic()-started})
            active=None
            _require(time.monotonic()-started<CAP_SECONDS,"unit GPU cap including failure reached")
            if index==4:
                _require(len(native_s_checks)==2,"native S destination not qualified for both probes")
                _write_new(out/"destination-s-check.json",{"matches":native_s_checks,
                    "destination":literal_binding(out/"destination-phase-raw.json")})
        _require(counts=={"model_forwards":11,"vision_forwards":1} and len(done)==10,
                 "finite eleven-call package incomplete")
        cost={"allocated_gpu_seconds":time.monotonic()-started,
              "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
              "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),**counts}
        _require(cost["allocated_gpu_seconds"]<CAP_SECONDS and
                 counts["model_forwards"]==11 and counts["vision_forwards"]==1 and
                 m["budget"]["sequence_cumulative_prior_gpu_hours"]+
                 cost["allocated_gpu_seconds"]/3600<8,
                 "cumulative GPU cap exceeded")
        pilot=_write_new(out/"pilot.json",{"schema":"recurrence_reverse_key_phase.pilot.v1",
              "status":"candidate_complete","protocol":literal_binding(PROTOCOL),
              "manifest":literal_binding(MANIFEST),"preflight":literal_binding(pre_path),
              "launch":launch,"effective_identity":identity,"input_identity":input_identity(batch),
              "source_planning":planning,"prefill":literal_binding(out/"prefill-raw.json"),
              "destination_phase":literal_binding(out/"destination-phase-raw.json"),
              "destination_s_check":literal_binding(out/"destination-s-check.json"),
              "phase_qualification":literal_binding(out/"phase-qualification.json"),
              "completed":done,"cost":cost})
        terminal={"status":"candidate_complete","terminal":True,"pilot":pilot,
                  "completed_suffix_cells":10,"cost":cost}
    except BaseException as exc:
        terminal={"status":"technical_invalid","terminal":True,"error":repr(exc),
                  "traceback":traceback.format_exc(),"completed":done,
                  "cost":{"allocated_gpu_seconds":time.monotonic()-started,
                          "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
                          "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),**counts}}
    finally:
        for hook in handles:hook.remove()
    terminal["artifact_bytes_before_receipt"]=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    _write_new(out/"receipt.json",terminal)
    if terminal["status"]!="candidate_complete":raise RuntimeError(terminal["error"])
    print(json.dumps({"status":terminal["status"],"cost":terminal["cost"]}))


def cold(out):
    m,_,native,_=contract()
    pre=json.loads((out/"preflight.json").read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and
             pre["manifest"]==literal_binding(MANIFEST) and
             pre["cpu_qualification"]==literal_binding(CPU_QUALIFICATION),
             "cold preflight changed")
    for item in pre["direct_source_captures"]:bound(item["maintained"]);bound(item["capture"])
    term=json.loads((out/"receipt.json").read_text())
    _require(term["status"]=="candidate_complete" and term["terminal"] and
             term["completed_suffix_cells"]==10,"cold terminal incomplete")
    pilot=json.loads(bound(term["pilot"]).read_text())
    _require(len(pilot["completed"])==10 and pilot["cost"]["model_forwards"]==11 and
             pilot["cost"]["vision_forwards"]==1,"cold call/cell count changed")
    prefill=json.loads(bound(pilot["prefill"]).read_text());bound(prefill["vector"])
    destination=json.loads(bound(pilot["destination_phase"]).read_text())
    bound(destination["vector"])
    checks=json.loads(bound(pilot["destination_s_check"]).read_text())
    _require(destination["positions"]==[
        PHASE["destination_earlier"]+PHASE["destination_latest"]]*3 and
        destination["fp64_full_destination_max_abs_error"]<=TOL and
        destination["earlier_matches_observed_latest_max_abs_error"]<=TOL and
        len(checks["matches"])==2 and all(x["max_abs_error"]<=TOL for x in checks["matches"]),
        "cold synthetic destination phase qualification changed")
    phase=json.loads(bound(pilot["phase_qualification"]).read_text())
    _require(len(phase["records"])==LAYERS and
             all(v["qualified"] for r in phase["records"] for k,v in r.items() if k!="layer"),
             "cold phase qualification changed")
    found=[]
    for item in pilot["completed"]:
        cell=json.loads(bound(item["record"]).read_text())
        vec=torch.load(bound(cell["vector"]),map_location="cpu",weights_only=True)
        observed=json.loads(bound(cell["consumer"]).read_text())
        _require(vec.ndim==1 and torch.isfinite(vec).all() and
                 int(vec.argmax())==cell["top2_ids"][0] and vec.numel()==cell["vocabulary_size"] and
                 math.isclose(float(torch.logsumexp(vec,-1)),cell["logsumexp"],abs_tol=1e-6) and
                 len(observed["before_attention"])==len(observed["after_attention"])==LAYERS and
                 len(observed["rotary_output_hashes"])==1 and not observed["media_present"],
                 "cold vector or all-layer actual consumer incomplete")
        owner,kind=cell["owner"],cell["kind"]
        refname="cross" if kind=="cross" else "native"
        original=json.loads(bound(m["references"][str(owner)][refname]["consumer"]).read_text())
        _require(observed["input_ids"]==original["input_ids"] and
                 observed["attention_mask"]==original["attention_mask"] and
                 observed["position_ids"]==original["position_ids"] and
                 observed["cache_position"]==list(range(WIDTH,FULL)),
                 "cold S history/position/physical slots changed")
        expected=observed["expected_historical_digest"]
        _require(len(expected)==LAYERS and all(
            observed["before_attention"][str(i)]["key_sha256"]==expected[i]["keys"] and
            observed["before_attention"][str(i)]["value_sha256"]==expected[i]["values"] and
            observed["before_attention"][str(i)]["value_sha256"]==prefill["native_segments"][i]["values"] and
            all(observed["before_attention"][str(i)]["selected_K_sha256"][name]==
                prefill["native_segments"][i][name]
                for name in (("earlier",) if kind in ("sham","latest") else
                             ("latest",) if kind=="earlier" else ("earlier","latest"))) and
            observed["after_attention"][str(i)]["historical_key_sha256"]==expected[i]["keys"] and
            observed["after_attention"][str(i)]["historical_value_sha256"]==expected[i]["values"]
            for i in range(LAYERS)),"cold actual historical K/V consumer differs")
        found.append((owner,kind))
    _require(found==plan(m),"cold ten-suffix order changed")
    return {"status":"passed","pilot":literal_binding(out/"pilot.json"),
            "terminal":literal_binding(out/"receipt.json"),"prefill":1,"suffix":10,
            "allocated_gpu_seconds":pilot["cost"]["allocated_gpu_seconds"]}


def tv(a,b):return .5*float(torch.sum(torch.abs(a-b)))


def reduce(out):
    m,_,_,_=contract();cold_result=cold(out)
    pilot=json.loads((out/"pilot.json").read_text())
    cells={(x["owner"],x["kind"]):json.loads(Path(x["record"]["path"]).read_text())
           for x in pilot["completed"]}
    result={}
    for p in m["probes"]:
        owner=p["owner"]
        raw={kind:torch.load(Path(cells[owner,kind]["vector"]["path"]),
                             map_location="cpu",weights_only=True).double()
             for kind in ("native","cross","sham","latest","earlier")}
        prob={k:torch.log_softmax(v,-1).exp() for k,v in raw.items()}
        B=tv(prob["cross"],prob["native"])
        rows={}
        for kind in ("latest","earlier"):
            error=tv(prob[kind],prob["cross"])
            rows[kind]={"tv_to_cross":error,"ratio_to_baseline":None if B<=1e-6 else error/B,
                        "tv_to_native":tv(prob[kind],prob["native"])}
        guard=m["decision"]["numerical_boundary_guard"]
        rlatest=rows["latest"]["ratio_to_baseline"]
        rearlier=rows["earlier"]["ratio_to_baseline"]
        near=B>1e-6 and (abs(rlatest-.5)<=guard or abs((rearlier-rlatest)-.1)<=guard)
        passing=B>1e-6 and not near and rlatest<.5 and rearlier-rlatest>.1
        category="unresolved_baseline" if B<=1e-6 else "numerical_HOLD" if near else "selective" if passing else "nonselective"
        pair=(152078,152083) if owner==1589003 else (152669,151670)
        result[str(owner)]={"baseline_tv":B,"treatments":rows,"earlier_minus_latest_ratio":
             None if B<=1e-6 else rearlier-rlatest,"category":category,
             "cells":{kind:{"winner":cells[owner,kind]["top2_ids"][0],
                            "runner":cells[owner,kind]["top2_ids"][1],
                            "gap":cells[owner,kind]["top2_gap"],
                            "fixed_pair_margin_logit":float(raw[kind][pair[0]]-raw[kind][pair[1]]),
                            "vector":cells[owner,kind]["vector"],
                            "consumer":cells[owner,kind]["consumer"]}
                      for kind in raw}}
    shared=all(x["category"]=="selective" for x in result.values())
    report={"schema":"recurrence_reverse_key_phase.reduction.v1","status":"candidate_complete",
            "protocol":literal_binding(PROTOCOL),"manifest":literal_binding(MANIFEST),
            "preflight":literal_binding(out/"preflight.json"),"pilot":literal_binding(out/"pilot.json"),
            "terminal":literal_binding(out/"receipt.json"),"cold_readback":cold_result,
            "results":result,"shared_strict_selective_sufficiency":shared,
            "decision":m["decision"],"cost":pilot["cost"],
            "cumulative_gpu_hours":m["budget"]["sequence_cumulative_prior_gpu_hours"]+
                                   pilot["cost"]["allocated_gpu_seconds"]/3600,
            "artifact_bytes_before_reduction":sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
    _write_new(out/"reduction.json",report)
    print(json.dumps({"status":report["status"],"shared_strict_selective_sufficiency":shared,
                      "probes":{k:{"B":v["baseline_tv"],"rows":v["treatments"],
                                   "category":v["category"]} for k,v in result.items()},
                      "cost":pilot["cost"]}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--output",type=Path,default=OUTPUT);parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="preflight":preflight(args.output)
    elif args.mode=="run":run(args.output,args.device)
    elif args.mode=="readback":
        value=cold(args.output);_write_new(args.output/"cold-readback.json",value);print(json.dumps(value))
    else:reduce(args.output)


if __name__=="__main__":main()
