"""One admitted four-request native-x1 relative-key-phase qualification."""

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
from transformers import AutoConfig, AutoTokenizer, DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import digest, literal_binding
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.native_row_choice.runtime import _trace_compare
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.numerical_feedback.select import token_hash
from probes.training_set_completion.recurrence_first_arrivals.prepare import _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest
from probes.training_set_completion.recurrence_native_x1_phase.run import (
    complex_rephase, qualify_phase,
)
from probes.training_set_completion.recurrence_position_history import swap_prefix_positions
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-cross-image-phase"
PROTOCOL = UNIT / "unit.md"
MANIFEST = UNIT / "manifest.json"
PROTOCOL_SHA = "30f310c3adb5e2be63a5616e36726d000527bc3fafdb83a91108ae481b7197eb"
MANIFEST_SHA = "1815e9e9cb5640876955d1f6b7b4bf2eb0d7d4ca4d292c60c2be374cef6201e0"
REPAIR = UNIT / "supporting/lead-repair-attempt-002.json"
REPAIR_SHA = "2f2af5ed04156338d5443fa30236a6635127b1eb6d8ad8b5e8d6dd30e7e748ad"
RED = UNIT / "supporting/attempt002-cpu-red.json"
GREEN = UNIT / "supporting/attempt002-cpu-green.json"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/attempt-002")
CASE = "mature:7511:5"
TARGET, SUFFIX, LAYERS, HEADS, DIM = 2, 5, 28, 8, 128
TOL = 2e-4
FAILED_SECONDS = 14.241613768041134
CAP_SECONDS = 540-FAILED_SECONDS
IMPORTS = [
    "probes/training_set_completion/recurrence_cross_image_phase/run.py",
    "probes/training_set_completion/recurrence_native_x1_phase/run.py",
    "probes/training_set_completion/recurrence_position_history.py",
    "probes/training_set_completion/recurrence_history_cache_partition.py",
    "probes/training_set_completion/recurrence_key_phase.py",
    "probes/training_set_completion/coordinate_continuity/runtime.py",
    "probes/training_set_completion/native_row_choice/runtime.py",
    "probes/training_set_completion/numerical_feedback/runtime.py",
    "probes/training_set_completion/numerical_feedback/select.py",
    "probes/training_set_completion/recurrence_first_arrivals/prepare.py",
    "probes/training_set_completion/recurrence_first_arrivals/stage1_case.py",
    "probes/training_set_completion/artifacts.py",
    "probes/training_set_completion/untied_shared.py",
    "src/artifacts/source_provenance.py", "src/qwen/input_identity.py",
    "src/qwen/native.py", "src/qwen/runtime_loading.py",
    "src/inference/bound_requests.py",
]


def bound(binding):
    path = Path(binding["path"])
    got = literal_binding(path)
    _require(got["sha256"] == binding["sha256"] and
             ("size_bytes" not in binding or got["size_bytes"] == binding["size_bytes"]),
             f"bound bytes changed: {path}")
    return path


def contract():
    _require(literal_binding(PROTOCOL)["sha256"] == PROTOCOL_SHA and
             literal_binding(MANIFEST)["sha256"] == MANIFEST_SHA, "lead contract changed")
    _require(literal_binding(REPAIR)["sha256"] == REPAIR_SHA,"lead repair ruling changed")
    repair=json.loads(REPAIR.read_text())
    failed=json.loads(bound(repair["failed_receipt"]).read_text())
    red=json.loads(RED.read_text())
    _require(repair["status"]=="lead-authorized-bounded-repair" and
             repair["budget"]["first_case_total_model_calls_including_failure"]==9 and
             repair["budget"]["first_case_total_vision_calls_including_failure"]==4 and
             repair["budget"]["first_case_remaining_seconds"]==CAP_SECONDS and
             failed["status"]=="technical_invalid" and
             failed["cost"]["allocated_gpu_seconds"]==FAILED_SECONDS and
             failed["cost"]["model_forwards"]==2 and failed["cost"]["vision_forwards"]==2 and
             red["status"]=="RED_expected_failure" and
             red["producer_sha256"]=="b6fbed754a32e863da825eb066f8d4eda5806d44e9dcefa3d9e7e983ced45a39" and
             red["repair_ruling_sha256"]==REPAIR_SHA and
             red["language_model_loads"]==red["cuda_calls"]==0 and
             "dict" in red["exception"],
             "failed attempt or repair budget changed")
    m = json.loads(MANIFEST.read_text())
    _require(m["status"] == "cohort-frozen-first-case-only-admitted" and
             m["admitted_case_ids"] == [CASE] and len(m["held_case_ids"]) == 7 and
             m["budget"]["currently_admitted_model_forwards"] == 7 and
             m["budget"]["currently_admitted_vision_forwards"] == 2 and
             m["budget"]["free_tokens"] == 0 and
             m["budget"]["first_case_gpu_hours_cap"] == .15 and
             m["budget"]["package_gpu_hours_cap"] == 1 and
             m["budget"]["sequence_gpu_hours_cap"] == 8 and
             m["decision"]["latest_ratio_max"] == .5 and
             m["decision"]["earlier_minus_latest_ratio_min"] == .1,
             "first-case admission changed")
    for key in ("registry", "predecessor_acceptance", "predecessor_manifest"):
        bound(m[key])
    registry = json.loads(Path(m["registry"]["path"]).read_text())
    _require([x["id"] for x in registry["selected"]] ==
             m["lead_cpu_selection_check"]["selected_ids"] and
             m["selected"] == registry["selected"] and
             m["effective_identity"] == registry["effective_identity"],
             "frozen eight-case registry changed")
    case = m["selected"][0]
    _require(case["id"] == CASE and case["batch_index"] == TARGET and
             case["antecedent_rows"] == [3,4] and case["current_row"] == 5 and
             case["geometry"]["first_y1_raw_offset"] == 50 and
             case["native_x1_token"] == 151670 and case["native_y1_token"] == 152209,
             "admitted native case changed")
    for key in ("raw", "trace", "runtime_receipt", "image"):
        bound(case["source_bindings"][key])
    receipt = json.loads(Path(case["source_bindings"]["runtime_receipt"]["path"]).read_text())
    _require(case["source_bindings"]["source_identity"] == registry["effective_identity"] and
             receipt["input_identity"]["request_ids"] ==
             [x["request_id"] for x in case["batch_companions"]],
             "four-request source identity changed")
    return m, case, receipt


def source_boundary(case, raw):
    b = case["source_bindings"]
    return {"group": case["group"], "batch_index": case["batch_index"],
            "image_id": case["image_id"], "raw_path": b["raw"]["path"],
            "trace_path": b["trace"]["path"],
            "receipt_path": b["runtime_receipt"]["path"],
            "native_tokens": raw[TARGET]["token_ids"],
            "native_token_hash": token_hash(raw[TARGET]["token_ids"])}


def source_inputs(model, batch, raw, case, pad):
    offset = case["geometry"]["first_y1_raw_offset"]
    tails = _prefix_tokens(raw, offset, pad)
    histories = [list(prompt)+tail for prompt,tail in zip(batch.prompt_token_ids,tails,strict=True)]
    full = exact_history_inputs(model,batch.inputs,histories,pad_token_id=pad,logits_to_keep=1)
    width = full["input_ids"].shape[1]
    full["cache_position"] = torch.arange(width,device=full["input_ids"].device)
    verify_full(full,case,raw)
    return full


def verify_full(full,case,raw):
    ids,mask,pos = (full[k] for k in ("input_ids","attention_mask","position_ids"))
    width = case["geometry"]["source_step_full_batch_width"]
    g = case["geometry"]
    _require(ids.shape == mask.shape == (4,width) and pos.shape == (3,4,width) and
             TARGET == case["batch_index"] and
             full["cache_position"].tolist() == list(range(width)),
             "source full-batch shape/index/slots changed")
    _require(ids[TARGET,-SUFFIX:].tolist() == raw[TARGET]["token_ids"][g["history_raw_offset"]:g["first_y1_raw_offset"]] and
             ids[TARGET,-1].item() == case["native_x1_token"] and
             raw[TARGET]["token_ids"][g["first_y1_raw_offset"]] == case["native_y1_token"] and
             g["physical_full_batch_padded"]["S"] == [width-SUFFIX,width] and
             pos[:,TARGET,-SUFFIX:].cpu().tolist() == g["rotary_position_ids"]["S"],
             "target native first-y1 prefix changed")
    for name in ("earlier","latest"):
        a,b = g["physical_full_batch_padded"][name]
        _require(b-a == 9 and pos[:,TARGET,a:b].cpu().tolist() == g["rotary_position_ids"][name],
                 f"{name} physical/rotary row changed")
    _require(all(mask[i,-SUFFIX:].tolist() == [1]*SUFFIX for i in range(4)),
             "source companion suffix masks changed")


def verify_cross(native,cross,target):
    _require(target == TARGET and native["input_ids"].shape[0] == 4 and
             torch.equal(native["input_ids"],cross["input_ids"]) and
             torch.equal(native["attention_mask"],cross["attention_mask"]) and
             torch.equal(native["position_ids"][:,:,:-SUFFIX],cross["position_ids"][:,:,:-SUFFIX]) and
             torch.equal(native["position_ids"][:,[i for i in range(4) if i!=target],-SUFFIX:],
                         cross["position_ids"][:,[i for i in range(4) if i!=target],-SUFFIX:]) and
             torch.equal(cross["position_ids"][:,target,-SUFFIX:],
                         native["position_ids"][:,target,-SUFFIX:]-9),
             "cross changed wrong target, content, companion, or phase sign")


def cpu_mutations(base,case,raw):
    verify_full(base,case,raw)
    donor = base["position_ids"][:,TARGET,-SUFFIX:]-9
    crossed = swap_prefix_positions(base,donor,TARGET,SUFFIX,tuple(base["input_ids"][TARGET,-SUFFIX:].tolist()))
    verify_cross(base,crossed,TARGET)
    checks=[]
    for label,target,mutation in (
        ("wrong_target_index",1,lambda x:x),
        ("wrong_S_sign",TARGET,lambda x:{**x,"position_ids":base["position_ids"]}),
        ("wrong_S_position",TARGET,lambda x:{**x,"position_ids":x["position_ids"].clone()}),
        ("companion_content",TARGET,lambda x:{**x,"input_ids":x["input_ids"].clone()}),
    ):
        changed = mutation(crossed)
        if label == "wrong_S_position": changed["position_ids"][:,TARGET,-1] += 1
        if label == "companion_content": changed["input_ids"][1,-1] += 1
        try: verify_cross(base,changed,target)
        except (ValueError,AssertionError): checks.append(label)
        else: raise AssertionError(f"CPU verifier missed {label}")
    bad = dict(case);bad["geometry"] = dict(case["geometry"])
    bad["geometry"]["physical_full_batch_padded"] = dict(case["geometry"]["physical_full_batch_padded"])
    bad["geometry"]["physical_full_batch_padded"]["earlier"] = [1348,1357]
    try: verify_full(base,bad,raw)
    except (ValueError,AssertionError): checks.append("wrong_historical_row_span")
    else: raise AssertionError("CPU verifier missed wrong historical row")
    return checks


class _ConfigOnlyRope:
    def __init__(self):
        self.config = AutoConfig.from_pretrained(BASE,local_files_only=True)
    get_rope_index = modeling_qwen3_vl.Qwen3VLModel.get_rope_index


def cost_forecast(m):
    acceptance=json.loads(bound(m["predecessor_acceptance"]).read_text())
    reduction=json.loads(bound(acceptance["reduction"]).read_text())
    prior=json.loads(bound(reduction["pilot"]).read_text())
    seconds=float(prior["cost"]["allocated_gpu_seconds"])
    _require(prior["cost"]["model_forwards"]==6 and prior["cost"]["vision_forwards"]==1,
             "prior measured route changed")
    reference_pixels=52*78*1536
    values=[]
    for case in m["selected"]:
        shape=case["pixel_elements_full_batch"]/reference_pixels
        width=case["geometry"]["source_step_full_batch_width"]/1421
        # Full native adds one model plus one vision pass; count it twice for planning.
        forecast=seconds*shape*width*(7/6)*2*2
        values.append({"id":case["id"],"pixel_ratio":shape,"width_ratio":width,
                       "seven_model_two_vision_seconds_with_2x_margin":forecast})
    first=values[0]["seven_model_two_vision_seconds_with_2x_margin"]
    package=sum(x["seven_model_two_vision_seconds_with_2x_margin"] for x in values)
    _require(first<CAP_SECONDS and FAILED_SECONDS+package<3600 and
             m["budget"]["sequence_cumulative_prior_gpu_hours"]+
             (FAILED_SECONDS+package)/3600<8,
             "new seven-call shape forecast exceeds admitted envelope")
    return {"prior_six_model_one_vision_seconds":seconds,"prior_pilot":acceptance["reduction"],
            "route_model_ratio":7/6,"second_vision_factor":2,"planning_margin":2,
            "per_case":values,"first_case_seconds":first,"package_seconds":package,
            "failed_attempt_seconds":FAILED_SECONDS,
            "first_case_remaining_cap_seconds":CAP_SECONDS,
            "package_cap_seconds_including_failure":3600,
            "sequence_prior_hours":m["budget"]["sequence_cumulative_prior_gpu_hours"]}


def preflight(out):
    m,case,receipt=contract()
    _require(not out.exists(),"attempt already exists")
    q=load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE),dtype="fp32",attn_implementation="sdpa",
        patch_embed_linearization="enabled",load_model=False))
    _require(q.model is None,"CPU preflight loaded model")
    raw=json.loads(Path(case["source_bindings"]["raw"]["path"]).read_text())["rows"]
    panel=json.loads((Path(case["source_bindings"]["raw"]["path"]).parents[3]/"panel.json").read_text())
    batch,raw,trace,group,planning=_source(source_boundary(case,raw),"untied",panel,q,torch.device("cpu"))
    _require(len(raw)==len(group["cases"])==4 and
             input_identity(batch)==receipt["input_identity"] and
             list(batch.request_ids)==case["source_bindings"]["input_identity_summary"]["request_ids"],
             "CPU original four-request identity changed")
    full=source_inputs(_ConfigOnlyRope(),batch,raw,case,int(q.tokenizer.pad_token_id))
    checks=cpu_mutations(full,case,raw)
    phase_cpu=phase_cpu_gate(case)
    saved_green=json.loads(GREEN.read_text())
    _require(saved_green["status"]=="passed" and
             saved_green["producer_sha256"]==literal_binding(Path(__file__))["sha256"] and
             saved_green["repair_ruling_sha256"]==REPAIR_SHA and
             saved_green["red_cpu_sha256"]==literal_binding(RED)["sha256"] and
             all(saved_green[key]==phase_cpu[key] for key in phase_cpu),
             "CPU repaired actual destination gate/source changed")
    forecast=cost_forecast(m)
    out.mkdir(parents=True)
    sources=[REPO/x for x in IMPORTS]
    sources += [Path(inspect.getfile(x)) for x in (cache_utils,modeling_qwen3_vl,sdpa_attention)]
    captures=[]
    for src in sources:
        rel=src.relative_to(REPO) if src.is_relative_to(REPO) else Path("transformers")/src.name
        saved=preserve_source(src,run_root=out,relative_name=rel)
        captures.append({"maintained":literal_binding(src),"capture":literal_binding(saved)})
    packet={"schema":"recurrence_cross_image_phase.preflight.v1","status":"cpu_qualified_before_gpu",
            "protocol":literal_binding(PROTOCOL),"manifest":literal_binding(MANIFEST),
            "repair_ruling":literal_binding(REPAIR),"failed_receipt":
            literal_binding(Path(json.loads(REPAIR.read_text())["failed_receipt"]["path"])),
            "red_cpu":literal_binding(RED),"green_cpu":literal_binding(GREEN),
            "registry":m["registry"],"admitted_case_id":CASE,
            "source":case["source_bindings"],"input_identity_sha256":digest(input_identity(batch)),
            "effective_identity_expected":receipt["identity"],
            "full_source_input_hashes":{k:tensor_hash(full[k]) for k in ("input_ids","attention_mask","position_ids","cache_position")},
            "full_source_shape":{"batch_size":4,"target_index":TARGET,"width":int(full["input_ids"].shape[1]),
                                 "image_grids":[list(g) for g in batch.image_grids],
                                 "pixel_elements":int(batch.inputs["pixel_values"].numel())},
            "cpu_checks":checks,"cpu_destination_phase":phase_cpu,
            "cost_forecast":forecast,"direct_source_captures":captures,
            "commands":{"gpu":["python","-B","-m","probes.training_set_completion.recurrence_cross_image_phase.run","run","--output",str(out),"--device","cuda:0"],
                        "readback":["python","-B","-m","probes.training_set_completion.recurrence_cross_image_phase.run","readback","--output",str(out)],
                        "reduce":["python","-B","-m","probes.training_set_completion.recurrence_cross_image_phase.run","reduce","--output",str(out)]}}
    _write_new(out/"preflight.json",packet)
    print(json.dumps({"status":packet["status"],"shape":packet["full_source_shape"],
                      "forecast_seconds":forecast["first_case_seconds"],
                      "package_forecast_seconds":forecast["package_seconds"],"captures":len(captures)}))


def checked_destination(native_positions, positions, expected_native):
    _require(set(native_positions)=={"earlier","latest"} and
             all(isinstance(native_positions[name],torch.Tensor) and
                 native_positions[name].shape==(3,9) and
                 torch.equal(native_positions[name].cpu(),torch.tensor(expected_native[name]))
                 for name in ("earlier","latest")),
             "native phase source/wrong historical span changed")
    expected=torch.cat((native_positions["earlier"]+9,native_positions["latest"]+9),dim=-1)
    _require(positions.shape==(3,18) and torch.equal(positions,expected) and
             all(torch.equal(positions[:,i*9:(i+1)*9]-native_positions[name],
                             torch.full_like(native_positions[name],9))
                 for i,name in enumerate(("earlier","latest"))),
             "destination sign, length, order or span changed")


def destination_phase(rotary, native_positions, device, expected_native):
    before={name:native_positions[name].clone() for name in ("earlier","latest")}
    shifted={name:native_positions[name]+9 for name in ("earlier","latest")}
    positions=torch.cat((shifted["earlier"],shifted["latest"]),dim=-1)
    checked_destination(native_positions,positions,expected_native)
    ids=positions.to(device).unsqueeze(1)
    with torch.no_grad():
        cos,sin=rotary(torch.empty((1,18,1),device=device,dtype=torch.float32),ids)
    _require(cos.shape==sin.shape==(1,18,DIM),"actual destination rotary shape changed")
    inv=rotary.inv_freq.detach().double().cpu();sections=list(rotary.mrope_section)
    _require(inv.shape==(DIM//2,) and len(sections)==3 and sum(sections)==DIM//2,
             "loaded rotary frequency/layout changed")
    axes=positions.double().cpu()[:,:,None]*inv[None,None,:]
    mixed=axes[0].clone()
    for axis in (1,2):
        indices=list(range(axis,3*sections[axis],3))
        mixed[:,indices]=axes[axis,:,indices]
    angle=torch.cat((mixed,mixed),dim=-1);scale=float(rotary.attention_scaling)
    err=max(float((cos[0].double().cpu()-angle.cos()*scale).abs().max()),
            float((sin[0].double().cpu()-angle.sin()*scale).abs().max()))
    _require(err<=TOL,"FP64 independent full destination phase mismatch")
    values={name:{"cos":cos[0,i*9:(i+1)*9].detach().float().cpu().clone(),
                  "sin":sin[0,i*9:(i+1)*9].detach().float().cpu().clone()}
            for i,name in enumerate(("destination_earlier","destination_latest"))}
    _require(all(torch.equal(native_positions[name],before[name]) for name in before),
             "destination construction mutated native position inputs")
    return values,{"positions":positions.tolist(),"inv_freq_sha256":tensor_hash(rotary.inv_freq),
                   "mrope_section":sections,"attention_scaling":scale,
                   "fp64_full_destination_max_abs_error":err,"rotary_module_invocations":1}


def phase_cpu_gate(case):
    native={name:torch.tensor(case["geometry"]["rotary_position_ids"][name],dtype=torch.long)
            for name in ("earlier","latest")}
    expected={name:value.tolist() for name,value in native.items()}
    old={name:value.clone() for name,value in native.items()}
    rotary=modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding(
        AutoConfig.from_pretrained(BASE,local_files_only=True).text_config,device="cpu")
    phases,record=destination_phase(rotary,native,torch.device("cpu"),expected)
    _require(record["positions"]==torch.cat((native["earlier"]+9,native["latest"]+9),-1).tolist() and
             all(phases[name][axis].shape==(9,DIM)
                 for name in ("destination_earlier","destination_latest") for axis in ("cos","sin")) and
             all(torch.equal(native[name],old[name]) for name in old) and
             record["fp64_full_destination_max_abs_error"]<=TOL,
             "CPU destination actual rotary/FP64/immutability check failed")
    good=torch.tensor(record["positions"],dtype=torch.long)
    mutations={"wrong_sign":torch.cat((native["earlier"]-9,native["latest"]-9),-1),
               "clipped_latest":good[:,:14],
               "wrong_span_order":torch.cat((native["latest"]+9,native["earlier"]+9),-1)}
    checks=[]
    for name,edited in mutations.items():
        try:checked_destination(native,edited,expected)
        except (ValueError,AssertionError):checks.append(name)
        else:raise AssertionError(f"CPU verifier missed {name}")
    wrong={"earlier":native["earlier"]+1,"latest":native["latest"]}
    try:checked_destination(wrong,good,expected)
    except (ValueError,AssertionError):checks.append("wrong_native_row_span")
    else:raise AssertionError("CPU verifier missed wrong native row span")
    return {"status":"passed","actual_destination_function":"destination_phase",
            "actual_rotary_module":"Qwen3VLTextRotaryEmbedding","device":"cpu",
            "native_positions":expected,"destination_positions":record["positions"],
            "phase_shapes":{name:{axis:list(value.shape) for axis,value in rec.items()}
                            for name,rec in phases.items()},
            "inputs_unchanged":True,"fp64_full_destination_max_abs_error":
            record["fp64_full_destination_max_abs_error"],"mutation_checks":checks,
            "language_model_loads":0,"cuda_calls":0}


@contextmanager
def patched(cache,kind,keys,digest_before,spans,width):
    """Patch, suffix forward, crop and restore in one inference scope."""
    with torch.inference_mode():
        saved=[]
        try:
            if kind not in ("native","cross"):
                row="latest" if kind=="sham" else kind
                a,b=spans[row]
                _require(kind in ("sham","latest","earlier") and b-a==9 and
                         len(keys)==LAYERS,"wrong patch kind/span/layers")
                for layer,key in zip(cache.layers,keys,strict=True):
                    _require(key.shape==(HEADS,9,DIM),"wrong candidate K shape")
                    old=layer.keys[TARGET,:,a:b,:].clone()
                    saved.append((layer,old))
                    layer.keys[TARGET,:,a:b,:].copy_(key)
            yield
        finally:
            cache.crop(width)
            if saved:
                a,b=spans["latest" if kind=="sham" else kind]
                for layer,old in saved:
                    layer.keys[TARGET,:,a:b,:].copy_(old)
            _require(cache.get_seq_length()==width and cache_digest(cache)==digest_before,
                     "historical cache failed finally restoration")


def run(out,device_name):
    started=time.monotonic()
    m,case,source_receipt=contract()
    pre=json.loads((out/"preflight.json").read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and pre["manifest"]==literal_binding(MANIFEST) and
             pre["repair_ruling"]==literal_binding(REPAIR) and
             pre["red_cpu"]==literal_binding(RED) and pre["green_cpu"]==literal_binding(GREEN) and
             pre["admitted_case_id"]==CASE and pre["cost_forecast"]["first_case_seconds"]<CAP_SECONDS and
             device_name=="cuda:0" and not (out/"receipt.json").exists(),
             "GPU first-case authority changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]);bound(item["capture"])
    launch=_write_new(out/"launch.json",{"status":"frozen_before_model_load","pid":os.getpid(),
              "started_unix":time.time(),"preflight":literal_binding(out/"preflight.json"),
              "producer":literal_binding(Path(__file__)),"hard_cap_seconds":CAP_SECONDS})
    device=torch.device(device_name);counts={"model_forwards":0,"vision_forwards":0}
    done=[];handles=[];active=None
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        q,identity=load_model("untied",device)
        saved=source_receipt["identity"]
        _require({k:v for k,v in identity.items() if k!="loader_source"}==
                 {k:v for k,v in saved.items() if k!="loader_source"} and
                 all(identity["loader_source"][k]==saved["loader_source"][k]
                     for k in ("sha256","size_bytes")),"loaded effective identity differs")
        model=q.model.eval()
        raw=json.loads(Path(case["source_bindings"]["raw"]["path"]).read_text())["rows"]
        panel=json.loads((Path(case["source_bindings"]["raw"]["path"]).parents[3]/"panel.json").read_text())
        batch,raw,trace,group,planning=_source(source_boundary(case,raw),"untied",panel,q,device)
        _require(len(raw)==len(group["cases"])==4 and
                 input_identity(batch)==source_receipt["input_identity"] and
                 digest(input_identity(batch))==pre["input_identity_sha256"],
                 "GPU full source batch identity changed")
        full=source_inputs(model,batch,raw,case,int(q.tokenizer.pad_token_id))
        _require({k:tensor_hash(full[k]) for k in pre["full_source_input_hashes"]}==
                 pre["full_source_input_hashes"],"GPU full source input/positions changed")
        width=full["input_ids"].shape[1]-SUFFIX;end=width+SUFFIX
        spans={k:tuple(case["geometry"]["physical_full_batch_padded"][k])
               for k in ("earlier","latest")}
        cross_positions=full["position_ids"][:,TARGET,width:end]-9
        cross=swap_prefix_positions(full,cross_positions,TARGET,SUFFIX,
                                    tuple(full["input_ids"][TARGET,width:end].tolist()))
        verify_cross(full,cross,TARGET)
        cache=DynamicCache()
        full.update(use_cache=False,return_dict=True,logits_to_keep=1)
        prefill=dict(full)
        prefill.update(input_ids=full["input_ids"][:,:width],
                       attention_mask=full["attention_mask"][:,:width],
                       position_ids=full["position_ids"][:,:,:width],
                       cache_position=torch.arange(width,device=device),
                       past_key_values=cache,use_cache=True)
        _require(prefill["input_ids"].shape==(4,width) and
                 torch.equal(prefill["input_ids"],full["input_ids"][:,:width]) and
                 torch.equal(prefill["attention_mask"],full["attention_mask"][:,:width]),
                 "native full-batch split changed")

        def before_model(_module,_args,kwargs):
            counts["model_forwards"]+=1
            _require(active is not None and counts["model_forwards"]<=7 and
                     time.monotonic()-started<CAP_SECONDS,"unexpected or over-cap model forward")
            for key in ("input_ids","attention_mask","position_ids","cache_position"):
                _require(torch.equal(kwargs.get(key),active["input"][key]),
                         f"actual {active['name']} model consumer {key} changed")
            media=active["name"] in ("full_native","prefill")
            _require(kwargs.get("use_cache") is (not (active["name"]=="full_native")) and
                     kwargs.get("past_key_values") is (None if active["name"]=="full_native" else cache),
                     "actual model cache route changed")
            for key in ("pixel_values","image_grid_thw"):
                _require((key in kwargs)==media,"actual media routing changed")
                if media:_require(tensor_hash(kwargs[key])==tensor_hash(active["input"][key]),
                                  "actual media values changed")
            active["observed"]={key:kwargs[key].detach().cpu().tolist()
                                for key in ("input_ids","attention_mask","position_ids","cache_position")}
            active["observed"].update(media_present=media,use_cache=kwargs["use_cache"],
                                      past_cache_present=kwargs.get("past_key_values") is not None)

        def before_vision(_module,_args):
            counts["vision_forwards"]+=1
            _require(counts["vision_forwards"]<=2,"vision forward ceiling reached")
        handles.extend((model.register_forward_pre_hook(before_model,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision)))

        # Call 1: original full source step, all four requests and images.
        active={"name":"full_native","input":full,"observed":{}}
        with torch.inference_mode():
            source_logits=model(**full).logits[:,-1,:].detach().float().cpu()
        torch.cuda.synchronize(device)
        _require(source_logits.shape[0]==4 and counts=={"model_forwards":1,"vision_forwards":1},
                 "full native source call incomplete")
        trace_parity=[_trace_compare(logits=source_logits[i],trace=trace,batch_index=i,
                                     absolute_offset=case["geometry"]["first_y1_raw_offset"],
                                     token_id=int(raw[i]["token_ids"][case["geometry"]["first_y1_raw_offset"]]),
                                     role="first_y1" if i==TARGET else "original_companion_step",atol=TOL)
                      for i in range(4)]
        _require(all(x["passed"] for x in trace_parity),"full native source/companion trace parity failed")
        full_dir=out/"cells/01-full-native";full_dir.mkdir(parents=True)
        torch.save(source_logits[TARGET].clone(),full_dir/"vocabulary.pt")
        torch.save(source_logits,full_dir/"full-batch-vocabulary.pt")
        full_record=_write_new(full_dir/"cell.json",{
            "kind":"full_native","vector":literal_binding(full_dir/"vocabulary.pt"),
            "full_batch_vector":literal_binding(full_dir/"full-batch-vocabulary.pt"),
            "consumer":_write_new(full_dir/"consumer-raw.json",active["observed"]),
            "trace_parity":trace_parity,"full_batch_top2_ids":
            [torch.topk(row,2).indices.tolist() for row in source_logits],
            "counts_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started})
        done.append({"kind":"full_native","record":full_record})

        # Call 2: cache the same physical historical prefix for all companions.
        rotary=model.model.language_model.rotary_emb
        seen_phase=[];pre_norm=[{} for _ in range(LAYERS)];capture=[]
        def on_prefill_rotary(_module,_args,output):
            cos,sin=output
            _require(cos.shape==sin.shape==(4,width,DIM),"prefill rotary shape changed")
            seen_phase.append({name:{"cos":cos[TARGET,a:b].detach().clone(),
                                     "sin":sin[TARGET,a:b].detach().clone()}
                               for name,(a,b) in spans.items()})
        capture.append(rotary.register_forward_hook(on_prefill_rotary))
        for index,layer in enumerate(model.model.language_model.layers):
            def on_pre_k(_module,_args,output,i=index):
                _require(output.shape==(4,width,HEADS,DIM),"normalized pre-K shape changed")
                pre_norm[i]={name:output[TARGET,a:b].transpose(0,1).detach().clone()
                             for name,(a,b) in spans.items()}
            capture.append(layer.self_attn.k_norm.register_forward_hook(on_pre_k))
        active={"name":"prefill","input":prefill,"observed":{}}
        try:
            with torch.inference_mode(): pref_out=model(**prefill)
        finally:
            for hook in capture:hook.remove()
        torch.cuda.synchronize(device)
        _require(pref_out.past_key_values is cache and len(seen_phase)==1 and
                 all(set(x)==set(spans) for x in pre_norm) and
                 counts=={"model_forwards":2,"vision_forwards":2},
                 "four-request prefill capture incomplete")
        _require(len(cache.layers)==LAYERS and all(
            isinstance(getattr(layer,axis),torch.Tensor) and
            getattr(layer,axis).shape==(4,HEADS,width,DIM) and
            getattr(layer,axis).dtype==torch.float32
            for layer in cache.layers for axis in ("keys","values")),
            "four-request K/V cache shape/dtype changed")
        native_digest=cache_digest(cache)
        native_segments=[{
            "earlier":tensor_hash(layer.keys[TARGET,:,spans["earlier"][0]:spans["earlier"][1],:]),
            "latest":tensor_hash(layer.keys[TARGET,:,spans["latest"][0]:spans["latest"][1],:]),
            "target_V":tensor_hash(layer.values[TARGET]),
            "other_K":tensor_hash(layer.keys[[i for i in range(4) if i!=TARGET]]),
            "other_V":tensor_hash(layer.values[[i for i in range(4) if i!=TARGET]])}
            for layer in cache.layers]
        native_positions={name:full["position_ids"][:,TARGET,a:b].cpu()
                          for name,(a,b) in spans.items()}
        destination,dest_record=destination_phase(
            rotary,native_positions,device,case["geometry"]["rotary_position_ids"])
        earlier_error=max(float((destination["destination_earlier"][axis]-
                                 seen_phase[0]["latest"][axis].cpu()).abs().max())
                          for axis in ("cos","sin"))
        _require(earlier_error<=TOL,"earlier destination/observed latest mismatch")
        dest_record["earlier_matches_latest_max_abs_error"]=earlier_error
        torch.save(destination,out/"destination-phase.pt")
        _write_new(out/"destination-phase-raw.json",{
            **dest_record,"vector":literal_binding(out/"destination-phase.pt")})
        seen_phase[0].update(destination)
        blocks={"phase":{name:{axis:value.detach().float().cpu() for axis,value in rec.items()}
                         for name,rec in seen_phase[0].items()},"layers":[]}
        for i,layer in enumerate(cache.layers):
            blocks["layers"].append({name:{
                "pre_k":pre_norm[i][name].float().cpu(),
                "native_k":layer.keys[TARGET,:,a:b,:].detach().float().cpu().clone(),
                "native_v":layer.values[TARGET,:,a:b,:].detach().float().cpu().clone()}
                for name,(a,b) in spans.items()})
        torch.save(blocks,out/"prefill-blocks.pt")
        _write_new(out/"prefill-raw.json",{
            "vector":literal_binding(out/"prefill-blocks.pt"),
            "consumer":active["observed"],"cache_digest":native_digest,
            "native_segments":native_segments,"counts_after":dict(counts),
            "allocated_gpu_seconds_after":time.monotonic()-started})
        phase_records,candidates=qualify_phase(blocks)
        _write_new(out/"phase-qualification.json",{
            "records":phase_records,"prefill_blocks":literal_binding(out/"prefill-blocks.pt")})
        _require(all(v["qualified"] for row in phase_records for k,v in row.items() if k!="layer"),
                 "all-layer scale-aware phase qualification failed")
        active=None

        native_vector=None;native_companions=None;sham_passed=False
        for step,kind in enumerate(("native","cross","sham","latest","earlier"),3):
            _require(step==3 or native_vector is not None,"cached native not qualified before cross")
            _require(step<=5 or sham_passed,"sham not qualified before treatments")
            parent=cross if kind=="cross" else full
            suffix={"input_ids":parent["input_ids"][:,width:end],
                    "attention_mask":parent["attention_mask"],
                    "position_ids":parent["position_ids"][:,:,width:end],
                    "cache_position":torch.arange(width,end,device=device),
                    "past_key_values":cache,"use_cache":True,"return_dict":True,"logits_to_keep":1}
            _require(suffix["input_ids"].shape==(4,SUFFIX) and
                     torch.equal(suffix["attention_mask"],full["attention_mask"]) and
                     torch.equal(suffix["input_ids"],full["input_ids"][:,width:end]),
                     "suffix companion/content input changed")
            edits=None if kind in ("native","cross") else candidates[kind]
            with patched(cache,kind,edits,native_digest,spans,width):
                expected=cache_digest(cache)
                for i,layer in enumerate(cache.layers):
                    _require(tensor_hash(layer.values[TARGET])==native_segments[i]["target_V"] and
                             tensor_hash(layer.keys[[j for j in range(4) if j!=TARGET]])==native_segments[i]["other_K"] and
                             tensor_hash(layer.values[[j for j in range(4) if j!=TARGET]])==native_segments[i]["other_V"],
                             "target V or companion historical K/V changed")
                    for row in ("earlier","latest"):
                        a,b=spans[row]
                        intended=(kind==row or kind=="sham" and row=="latest")
                        required=tensor_hash(edits[i]) if intended else native_segments[i][row]
                        _require(tensor_hash(layer.keys[TARGET,:,a:b,:])==required,
                                 "selected historical K differs from treatment/sham")
                active={"name":kind,"input":suffix,"observed":{}}
                before_rows={};after_rows={};rotary_record=[];rotary_values=[];local=[]
                def on_suffix_rotary(_module,args,output):
                    _require(torch.equal(args[1],suffix["position_ids"]),"actual S rotary input changed")
                    cos,sin=output
                    _require(cos.shape==sin.shape==(4,SUFFIX,DIM),"actual four-request S rotary shape changed")
                    rotary_record.append({"cos":tensor_hash(cos),"sin":tensor_hash(sin)})
                    rotary_values.append((cos,sin))
                local.append(rotary.register_forward_hook(on_suffix_rotary))
                for index,layer in enumerate(model.model.language_model.layers):
                    def before_attention(_module,_args,kwargs,i=index):
                        _require(kwargs.get("past_key_values") is cache and
                                 cache.get_seq_length(i)==width and
                                 torch.equal(kwargs.get("cache_position"),suffix["cache_position"]),
                                 "actual attention cache slots changed")
                        mask=kwargs.get("attention_mask")
                        _require(isinstance(mask,torch.Tensor) and mask.dtype==torch.bool and
                                 mask.shape==(4,1,SUFFIX,end),"actual attention mask shape changed")
                        causal=torch.arange(end,device=mask.device)[None,:] <= (
                            width+torch.arange(SUFFIX,device=mask.device)[:,None])
                        allowed=causal[None,:,:] & suffix["attention_mask"][:,:,None].transpose(1,2).bool()
                        _require(torch.equal(mask[:,0],allowed),"actual companion/padding causal mask changed")
                        embed=kwargs.get("position_embeddings")
                        _require(len(rotary_record)==1 and isinstance(embed,tuple) and len(embed)==2 and
                                 torch.equal(embed[0],rotary_values[0][0]) and
                                 torch.equal(embed[1],rotary_values[0][1]),
                                 "actual rotary embedding consumer changed")
                        current=cache.layers[i]
                        observed={axis:tensor_hash(getattr(current,axis)) for axis in ("keys","values")}
                        _require(observed==expected[i],"attention consumed wrong historical cache")
                        before_rows[i]={"key_sha256":observed["keys"],
                                        "value_sha256":observed["values"],
                                        "mask_sha256":tensor_hash(mask),
                                        "selected_K_sha256":{name:tensor_hash(current.keys[TARGET,:,a:b,:])
                                                             for name,(a,b) in spans.items()},
                                        "target_V_sha256":tensor_hash(current.values[TARGET]),
                                        "companion_K_sha256":tensor_hash(current.keys[[j for j in range(4) if j!=TARGET]]),
                                        "companion_V_sha256":tensor_hash(current.values[[j for j in range(4) if j!=TARGET]])}
                    def before_output(_module,_args,i=index):
                        _require(cache.get_seq_length(i)==end,"S keys not appended before attention output")
                        current=cache.layers[i]
                        _require({axis:tensor_hash(getattr(current,axis)[:,:,:width,:])
                                  for axis in ("keys","values")}==expected[i],
                                 "historical cache mutated during suffix")
                        after_rows[i]={"companion_suffix_K_sha256":
                                       tensor_hash(current.keys[[j for j in range(4) if j!=TARGET],:,width:end,:]),
                                       "companion_suffix_V_sha256":
                                       tensor_hash(current.values[[j for j in range(4) if j!=TARGET],:,width:end,:]),
                                       "historical_key_sha256":expected[i]["keys"],
                                       "historical_value_sha256":expected[i]["values"]}
                    local.extend((layer.self_attn.register_forward_pre_hook(before_attention,with_kwargs=True),
                                  layer.self_attn.o_proj.register_forward_pre_hook(before_output)))
                try:
                    with torch.inference_mode():logits=model(**suffix).logits[:,-1,:].detach().float().cpu()
                finally:
                    for hook in local:hook.remove()
                torch.cuda.synchronize(device)
                _require(logits.shape[0]==4 and len(rotary_record)==1 and
                         len(before_rows)==len(after_rows)==LAYERS and
                         counts["vision_forwards"]==2,"actual full-batch suffix consumer incomplete")
                if kind=="native":
                    err=max(float((rotary_values[0][j][TARGET].detach().cpu()-
                                   destination["destination_latest"][axis][:SUFFIX]).abs().max())
                            for j,axis in enumerate(("cos","sin")))
                    _require(err<=TOL,"latest destination first five differ from actual native S")
                    _write_new(out/"destination-s-check.json",{
                        "max_abs_error":err,"destination":literal_binding(out/"destination-phase-raw.json")})
                companions=[{key:after_rows[i][key] for key in
                             ("companion_suffix_K_sha256","companion_suffix_V_sha256")}
                            for i in range(LAYERS)]
                if kind=="native":native_companions=companions
                else:_require(companions==native_companions,
                              "companion S K/V changed under target-only intervention")
                cell_dir=out/"cells"/f"{step:02d}-{kind}";cell_dir.mkdir(parents=True)
                vec=logits[TARGET].clone();torch.save(vec,cell_dir/"vocabulary.pt")
                if kind=="native":torch.save(logits,cell_dir/"full-batch-vocabulary.pt")
                consumer=_write_new(cell_dir/"consumer-raw.json",{
                    **active["observed"],"rotary_output_hashes":rotary_record,
                    "before_attention":before_rows,"after_attention":after_rows,
                    "expected_historical_digest":expected})
                _require(vec.ndim==1 and torch.isfinite(vec).all(),"invalid target vocabulary vector")
                top=torch.topk(vec,2)
                record={"kind":kind,"vector":literal_binding(cell_dir/"vocabulary.pt"),
                        "consumer":consumer,"top2_ids":top.indices.tolist(),
                        "top2_logits":top.values.tolist(),
                        "top2_gap":float(top.values[0]-top.values[1]),
                        "logsumexp":float(torch.logsumexp(vec,-1)),
                        "native_y1_log_probability":float(torch.log_softmax(vec.double(),-1)[case["native_y1_token"]]),
                        "vocabulary_size":int(vec.numel()),
                        "companion_top2_ids":[torch.topk(logits[i],2).indices.tolist()
                                              for i in range(4) if i!=TARGET],
                        "counts_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started}
                if kind=="native":
                    record["full_batch_vector"]=literal_binding(cell_dir/"full-batch-vocabulary.pt")
                    errors=[float((logits[i]-source_logits[i]).abs().max()) for i in range(4)]
                    _require(all(v<=TOL for v in errors),"cached/full native or companion vector mismatch")
                    record["full_native_max_abs_error_by_batch_index"]=errors
                    native_vector=vec.clone()
                elif kind=="sham":
                    error=float((vec-native_vector).abs().max())
                    _require(error<=TOL,"latest identity sham differs from cached native")
                    record["native_sham_max_abs_error"]=error;sham_passed=True
                elif kind=="cross":record["vector_reference_status"]="new_cross_readout"
                cell=_write_new(cell_dir/"cell.json",record)
                done.append({"kind":kind,"record":cell})
                _write_new(out/f"checkpoint-{step:02d}.json",{
                    "completed":done,"counts":dict(counts),
                    "allocated_gpu_seconds":time.monotonic()-started})
            active=None
            _require(time.monotonic()-started<CAP_SECONDS,"first-case cap including setup reached")
        _require(counts=={"model_forwards":7,"vision_forwards":2} and len(done)==6,
                 "finite first-case calls incomplete")
        cost={"allocated_gpu_seconds":time.monotonic()-started,
              "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
              "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),**counts}
        _require(cost["allocated_gpu_seconds"]<CAP_SECONDS and
                 FAILED_SECONDS+cost["allocated_gpu_seconds"]<540 and
                 FAILED_SECONDS+cost["allocated_gpu_seconds"]<3600 and
                 m["budget"]["sequence_cumulative_prior_gpu_hours"]+
                 (FAILED_SECONDS+cost["allocated_gpu_seconds"])/3600<8 and
                 counts["model_forwards"]+2==9 and counts["vision_forwards"]+2==4,
                 "first-case/package/sequence GPU cap exceeded")
        pilot=_write_new(out/"pilot.json",{
            "schema":"recurrence_cross_image_phase.pilot.v1","status":"candidate_complete",
            "protocol":literal_binding(PROTOCOL),"manifest":literal_binding(MANIFEST),
            "repair_ruling":literal_binding(REPAIR),"failed_receipt":pre["failed_receipt"],
            "preflight":literal_binding(out/"preflight.json"),"launch":launch,
            "effective_identity":identity,"input_identity_sha256":digest(input_identity(batch)),
            "source_planning_replanned":planning.get("replanned_image_plan"),
            "full_native":done[0]["record"],"prefill":literal_binding(out/"prefill-raw.json"),
            "destination_phase":literal_binding(out/"destination-phase-raw.json"),
            "destination_s_check":literal_binding(out/"destination-s-check.json"),
            "phase_qualification":literal_binding(out/"phase-qualification.json"),
            "completed":done,"cost":cost,
            "first_case_total_model_forwards_including_failure":9,
            "first_case_total_vision_forwards_including_failure":4,
            "first_case_total_gpu_seconds_including_failure":
            FAILED_SECONDS+cost["allocated_gpu_seconds"]})
        terminal={"status":"candidate_complete","terminal":True,"pilot":pilot,
                  "completed_cells":len(done),"cost":cost,
                  "failed_attempt_seconds":FAILED_SECONDS,
                  "first_case_total_gpu_seconds":FAILED_SECONDS+cost["allocated_gpu_seconds"]}
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
    m,case,receipt=contract()
    pre=json.loads((out/"preflight.json").read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and
             pre["manifest"]==literal_binding(MANIFEST) and
             pre["repair_ruling"]==literal_binding(REPAIR) and
             pre["red_cpu"]==literal_binding(RED) and pre["green_cpu"]==literal_binding(GREEN) and
             pre["admitted_case_id"]==CASE,"cold preflight changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]);bound(item["capture"])
    term=json.loads((out/"receipt.json").read_text())
    _require(term["status"]=="candidate_complete" and term["terminal"] and
             term["completed_cells"]==6 and
             term["cost"]["model_forwards"]==7 and term["cost"]["vision_forwards"]==2 and
             term["failed_attempt_seconds"]==FAILED_SECONDS and
             term["first_case_total_gpu_seconds"]<540,
             "cold terminal calls incomplete")
    pilot=json.loads(bound(term["pilot"]).read_text())
    identity,saved=pilot["effective_identity"],receipt["identity"]
    _require(pilot["preflight"]==literal_binding(out/"preflight.json") and
             pilot["repair_ruling"]==literal_binding(REPAIR) and
             pilot["failed_receipt"]==pre["failed_receipt"] and
             pilot["first_case_total_model_forwards_including_failure"]==9 and
             pilot["first_case_total_vision_forwards_including_failure"]==4 and
             {k:v for k,v in identity.items() if k!="loader_source"}==
             {k:v for k,v in saved.items() if k!="loader_source"} and
             all(identity["loader_source"][k]==saved["loader_source"][k]
                 for k in ("sha256","size_bytes")) and
             pilot["input_identity_sha256"]==pre["input_identity_sha256"] and
             [x["kind"] for x in pilot["completed"]]==
             ["full_native","native","cross","sham","latest","earlier"],
             "cold provenance/call order changed")
    full=json.loads(bound(pilot["full_native"]).read_text())
    consumer=json.loads(bound(full["consumer"]).read_text())
    full_vectors=torch.load(bound(full["full_batch_vector"]),map_location="cpu",weights_only=True)
    _require(full_vectors.ndim==2 and full_vectors.shape[0]==4 and
             torch.equal(full_vectors[TARGET],torch.load(bound(full["vector"]),map_location="cpu",weights_only=True)) and
             all(x["passed"] for x in full["trace_parity"]) and
             len(full["trace_parity"])==4 and consumer["media_present"] and
             not consumer["use_cache"] and not consumer["past_cache_present"],
             "cold full native source vector/consumer changed")
    raw=json.loads(Path(case["source_bindings"]["raw"]["path"]).read_text())["rows"]
    offset=case["geometry"]["first_y1_raw_offset"]
    prompts=receipt["input_identity"]["prompt_token_ids"]
    pad=int(AutoTokenizer.from_pretrained(BASE,local_files_only=True).pad_token_id)
    # Source receipt has exact unpadded prompts. The first case has no ended
    # companion before step 50, so no source pad token is inserted in tails.
    _require(all(len(x["token_ids"])>offset for x in raw),"cold companion reached EOS before source step")
    histories=[list(p)+list(r["token_ids"][:offset]) for p,r in zip(prompts,raw,strict=True)]
    from src.qwen.native import padded_histories
    ids,mask=padded_histories(histories,pad_token_id=pad)
    rope=_ConfigOnlyRope()
    grid=torch.tensor(receipt["input_identity"]["image_grids"],dtype=torch.long)
    positions,_=rope.get_rope_index(ids,grid,None,mask)
    _require(consumer["input_ids"]==ids.tolist() and
             consumer["attention_mask"]==mask.tolist() and
             consumer["position_ids"]==positions.tolist() and
             consumer["cache_position"]==list(range(ids.shape[1])) and
             {"input_ids":tensor_hash(ids),"attention_mask":tensor_hash(mask),
              "position_ids":tensor_hash(positions),
              "cache_position":tensor_hash(torch.arange(ids.shape[1]))}==
             pre["full_source_input_hashes"],
             "cold four-request full source input/rotary differs")
    prefill=json.loads(bound(pilot["prefill"]).read_text())
    blocks=torch.load(bound(prefill["vector"]),map_location="cpu",weights_only=True)
    phase=json.loads(bound(pilot["phase_qualification"]).read_text())
    _require(len(phase["records"])==LAYERS and
             all(v["qualified"] for row in phase["records"] for k,v in row.items() if k!="layer"),
             "cold all-layer numerical phase gate changed")
    rebuilt_records,candidates=qualify_phase(blocks)
    _require(rebuilt_records==phase["records"],"cold phase qualification recomputation differs")
    dest=json.loads(bound(pilot["destination_phase"]).read_text())
    bound(dest["vector"])
    scheck=json.loads(bound(pilot["destination_s_check"]).read_text())
    _require(dest["fp64_full_destination_max_abs_error"]<=TOL and
             dest["earlier_matches_latest_max_abs_error"]<=TOL and
             scheck["max_abs_error"]<=TOL,"cold destination/actual S phase changed")
    cells={x["kind"]:json.loads(bound(x["record"]).read_text()) for x in pilot["completed"]}
    vectors={kind:torch.load(bound(cell["vector"]),map_location="cpu",weights_only=True)
             for kind,cell in cells.items()}
    native_full=torch.load(bound(cells["native"]["full_batch_vector"]),map_location="cpu",weights_only=True)
    native_error=(native_full-full_vectors).abs().amax(dim=1).tolist()
    _require(max(native_error)<=TOL and
             all(abs(x-y)<=1e-7 for x,y in zip(native_error,cells["native"]["full_native_max_abs_error_by_batch_index"])) and
             torch.equal(native_full[TARGET],vectors["native"]) and
             float((vectors["native"]-vectors["sham"]).abs().max())<=TOL,
             "cold cached/full or sham agreement changed")
    width=ids.shape[1]-SUFFIX
    native_companion_suffix=None
    for kind in ("native","cross","sham","latest","earlier"):
        cell=cells[kind];obs=json.loads(bound(cell["consumer"]).read_text());vec=vectors[kind]
        _require(vec.ndim==1 and vec.numel()==full_vectors.shape[-1] and torch.isfinite(vec).all() and
                 torch.topk(vec,2).indices.tolist()==cell["top2_ids"] and
                 math.isclose(float(torch.logsumexp(vec,-1)),cell["logsumexp"],abs_tol=1e-6) and
                 math.isclose(float(torch.log_softmax(vec.double(),-1)[case["native_y1_token"]]),
                              cell["native_y1_log_probability"],abs_tol=1e-6),
                 f"cold {kind} full vocabulary summary changed")
        expected_pos=positions[:,:,width:].clone()
        if kind=="cross":expected_pos[:,TARGET]-=9
        _require(obs["input_ids"]==ids[:,width:].tolist() and
                 obs["attention_mask"]==mask.tolist() and
                 obs["position_ids"]==expected_pos.tolist() and
                 obs["cache_position"]==list(range(width,ids.shape[1])) and
                 not obs["media_present"] and obs["use_cache"] and obs["past_cache_present"] and
                 len(obs["before_attention"])==len(obs["after_attention"])==LAYERS and
                 len(obs["rotary_output_hashes"])==1,
                 f"cold {kind} actual S consumer changed")
        companion_suffix=[{key:obs["after_attention"][str(i)][key] for key in
                           ("companion_suffix_K_sha256","companion_suffix_V_sha256")}
                          for i in range(LAYERS)]
        if kind=="native":native_companion_suffix=companion_suffix
        else:_require(companion_suffix==native_companion_suffix,
                      f"cold {kind} companion suffix K/V changed")
        expected=obs["expected_historical_digest"]
        _require(len(expected)==LAYERS and all(
            obs["before_attention"][str(i)]["key_sha256"]==expected[i]["keys"] and
            obs["before_attention"][str(i)]["value_sha256"]==expected[i]["values"] and
            obs["after_attention"][str(i)]["historical_key_sha256"]==expected[i]["keys"] and
            obs["after_attention"][str(i)]["historical_value_sha256"]==expected[i]["values"] and
            obs["before_attention"][str(i)]["target_V_sha256"]==prefill["native_segments"][i]["target_V"] and
            obs["before_attention"][str(i)]["companion_K_sha256"]==prefill["native_segments"][i]["other_K"] and
            obs["before_attention"][str(i)]["companion_V_sha256"]==prefill["native_segments"][i]["other_V"] and
            all(obs["before_attention"][str(i)]["selected_K_sha256"][row]==
                (tensor_hash(candidates[kind][i]) if kind==row or kind=="sham" and row=="latest"
                 else prefill["native_segments"][i][row]) for row in ("earlier","latest"))
            for i in range(LAYERS)),f"cold {kind} all-layer actual historical K/V changed")
    return {"status":"passed","pilot":literal_binding(out/"pilot.json"),
            "terminal":literal_binding(out/"receipt.json"),"calls":7,"vision":2,
            "full_batch_native_max_abs_errors":native_error,
            "allocated_gpu_seconds":term["cost"]["allocated_gpu_seconds"]}


def tv(p,q):return .5*float(torch.sum(torch.abs(p-q)))


def reduce(out):
    m,case,_=contract()
    readback=cold(out)
    pilot=json.loads((out/"pilot.json").read_text())
    cells={x["kind"]:json.loads(Path(x["record"]["path"]).read_text())
           for x in pilot["completed"]}
    logits={kind:torch.load(Path(cell["vector"]["path"]),map_location="cpu",weights_only=True).double()
            for kind,cell in cells.items()}
    prob={kind:torch.log_softmax(value,-1).exp() for kind,value in logits.items()}
    baseline=tv(prob["native"],prob["cross"])
    treatments={}
    for kind in ("latest","earlier"):
        error=tv(prob[kind],prob["cross"])
        treatments[kind]={"tv_to_cross":error,"ratio_to_baseline":None if baseline<=1e-6 else error/baseline,
                          "tv_to_native":tv(prob[kind],prob["native"])}
    latest=treatments["latest"]["ratio_to_baseline"]
    earlier=treatments["earlier"]["ratio_to_baseline"]
    difference=None if baseline<=1e-6 else earlier-latest
    guard=m["decision"]["numerical_guard"]
    near=baseline>1e-6 and (abs(latest-.5)<=guard or abs(difference-.1)<=guard)
    passed=baseline>1e-6 and not near and latest<.5 and difference>.1
    category=("nondiscriminating_baseline" if baseline<=1e-6 else
              "numerical_HOLD" if near else "selective_transfer" if passed else "nonselective_transfer")
    summary={"schema":"recurrence_cross_image_phase.first_case_reduction.v1",
             "status":"candidate","case_id":CASE,"category":category,
             "decision":m["decision"],"baseline_tv":baseline,"treatments":treatments,
             "earlier_minus_latest_ratio":difference,
             "cells":{kind:{"winner":cell["top2_ids"][0] if "top2_ids" in cell else int(logits[kind].argmax()),
                            "runner":cell["top2_ids"][1] if "top2_ids" in cell else int(torch.topk(logits[kind],2).indices[1]),
                            "gap":cell["top2_gap"] if "top2_gap" in cell else float(torch.topk(logits[kind],2).values.diff().abs()[0]),
                            "native_y1_log_probability":float(torch.log_softmax(logits[kind],-1)[case["native_y1_token"]]),
                            "vector":cell["vector"]}
                      for kind,cell in cells.items()},
             "cold_readback":readback,"cost":pilot["cost"],
             "measured_reforecast_status":"remaining_seven_HOLD_lead_review"}
    _write_new(out/"reduction.json",summary)
    print(json.dumps({"category":category,"baseline_tv":baseline,
                      "latest_ratio":latest,"earlier_ratio":earlier,
                      "first_case_gpu_seconds":pilot["cost"]["allocated_gpu_seconds"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="preflight":preflight(args.output)
    elif args.mode=="run":run(args.output,args.device)
    elif args.mode=="readback":print(json.dumps(cold(args.output)))
    else:reduce(args.output)


if __name__=="__main__":main()
