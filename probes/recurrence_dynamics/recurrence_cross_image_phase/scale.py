"""Seven frozen four-request native-x1 relative-key-phase qualifications."""

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

from src.artifacts.utf8_json import digest, literal_binding
from probes.model_profiles.mature_source import load_saved_source as _source
from src.qwen.native_row_scores import compare_saved_trace as _trace_compare
from src.qwen.saved_prefix import prefix_tokens as _prefix_tokens
from probes.recurrence_dynamics.numerical_feedback.select import token_hash
from probes.recurrence_dynamics.recurrence_first_arrivals.prepare import _require
from probes.recurrence_dynamics.recurrence_first_arrivals.stage1_case import _write_new
from probes.recurrence_dynamics.recurrence_history_cache_partition import cache_digest
from probes.recurrence_dynamics.recurrence_native_x1_phase.run import complex_rephase, qualify_phase
from probes.recurrence_dynamics.recurrence_position_history import swap_prefix_positions
from probes.recurrence_dynamics.recurrence_cross_image_phase.run import destination_phase, phase_cpu_gate, _ConfigOnlyRope, tv
from probes.model_profiles.mature_tied_untied import BASE, load_model
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
ADMISSION = UNIT / "lead-remaining-admission-v1.json"
ADMISSION_SHA = "74fd8ede6820b0f1c27c2c5f8008343605e27f554d5ab4cbceb9957cb51f96e4"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1")
CASE_IDS = ("mature:14038:10", "mature:351017:5", "mature:99184:7",
            "mature:1584:2", "mature:2299:2", "mature:2685:12", "mature:4134:7")
SUFFIX, LAYERS, HEADS, DIM = 5, 28, 8, 128
TOL = 2e-4
PRIOR_SECONDS = 64.7263178229332
IMPORTS = [
    'probes/recurrence_dynamics/recurrence_cross_image_phase/scale.py',
    'probes/recurrence_dynamics/recurrence_cross_image_phase/run.py',
    'probes/recurrence_dynamics/recurrence_native_x1_phase/run.py',
    'probes/recurrence_dynamics/recurrence_position_history.py',
    'probes/recurrence_dynamics/recurrence_history_cache_partition.py',
    'probes/recurrence_dynamics/recurrence_key_phase.py',
    'probes/recurrence_dynamics/coordinate_continuity/runtime.py',
    'probes/recurrence_dynamics/native_row_choice/runtime.py',
    'probes/recurrence_dynamics/numerical_feedback/runtime.py',
    'probes/recurrence_dynamics/numerical_feedback/select.py',
    'probes/recurrence_dynamics/recurrence_first_arrivals/prepare.py',
    'probes/recurrence_dynamics/recurrence_first_arrivals/stage1_case.py',
    'src/artifacts/utf8_json.py',
    "probes/model_profiles/mature_tied_untied.py",
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


def contract(case_id):
    _require(literal_binding(PROTOCOL)["sha256"] == PROTOCOL_SHA and
             literal_binding(MANIFEST)["sha256"] == MANIFEST_SHA and
             literal_binding(ADMISSION)["sha256"] == ADMISSION_SHA,
             "lead contract changed")
    admission = json.loads(ADMISSION.read_text())
    _require(admission["status"] == "lead-admitted-remaining-seven" and
             tuple(admission["case_ids"]) == CASE_IDS and
             admission["budget"]["additional_model_forwards_cap"] == 49 and
             admission["budget"]["additional_vision_forwards_cap"] == 14 and
             admission["budget"]["charged_package_gpu_seconds"] == PRIOR_SECONDS,
             "seven-case admission changed")
    for key in ("original_protocol", "original_manifest", "first_case_acceptance",
                "frozen_r1_producer", "first_case_candidate"):
        bound(admission[key])
    m = json.loads(MANIFEST.read_text())
    _require(m["status"] == "cohort-frozen-first-case-only-admitted" and
             [x["id"] for x in m["selected"][1:]] == list(CASE_IDS) and
             m["decision"]["latest_ratio_max"] == .5 and
             m["decision"]["earlier_minus_latest_ratio_min"] == .1 and
             m["budget"]["package_gpu_hours_cap"] == 1 and
             m["budget"]["sequence_gpu_hours_cap"] == 8,
             "frozen selection or criteria changed")
    for key in ("registry", "predecessor_acceptance", "predecessor_manifest"):
        bound(m[key])
    registry = json.loads(Path(m["registry"]["path"]).read_text())
    _require(m["selected"] == registry["selected"] and
             m["effective_identity"] == registry["effective_identity"],
             "frozen registry changed")
    _require(case_id in CASE_IDS, "case outside admitted queue")
    case = m["selected"][1 + CASE_IDS.index(case_id)]
    for key in ("raw", "trace", "runtime_receipt", "image"):
        bound(case["source_bindings"][key])
    receipt = json.loads(Path(case["source_bindings"]["runtime_receipt"]["path"]).read_text())
    _require(case["source_bindings"]["source_identity"] == registry["effective_identity"] and
             receipt["input_identity"]["request_ids"] ==
             [x["request_id"] for x in case["batch_companions"]] and
             case["batch_index"] in range(4), "source identity/target index changed")
    return m, case, receipt, admission

def source_boundary(case, raw):
    b = case["source_bindings"]
    return {"group": case["group"], "batch_index": case["batch_index"],
            "image_id": case["image_id"], "raw_path": b["raw"]["path"],
            "trace_path": b["trace"]["path"],
            "receipt_path": b["runtime_receipt"]["path"],
            "native_tokens": raw[case["batch_index"]]["token_ids"],
            "native_token_hash": token_hash(raw[case["batch_index"]]["token_ids"])}


def source_inputs(model, batch, raw, case, pad):
    offset = case["geometry"]["first_y1_raw_offset"]
    tails = _prefix_tokens(raw, offset, pad)
    histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
    full = exact_history_inputs(model, batch.inputs, histories, pad_token_id=pad, logits_to_keep=1)
    width = full["input_ids"].shape[1]
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    verify_full(full, case, raw, batch.prompt_token_ids, pad)
    return full


def verify_full(full, case, raw, prompts, pad):
    ids, mask, pos = (full[k] for k in ("input_ids", "attention_mask", "position_ids"))
    g = case["geometry"]; width = g["source_step_full_batch_width"]
    target = case["batch_index"]; offset = g["first_y1_raw_offset"]
    tails = _prefix_tokens(raw, offset, pad)
    _require(ids.shape == mask.shape == (4, width) and pos.shape == (3, 4, width) and
             full["cache_position"].tolist() == list(range(width)) and
             len(prompts) == len(raw) == 4, "source full-batch shape/slots changed")
    for i, (prompt, tail) in enumerate(zip(prompts, tails, strict=True)):
        expected = list(prompt) + tail
        actual = ids[i, -len(expected):].tolist()
        _require(actual == expected and mask[i, -len(expected):].tolist() == [1] * len(expected) and
                 mask[i, :width-len(expected)].tolist() == [0] * (width-len(expected)) and
                 ids[i, :width-len(expected)].tolist() == [pad] * (width-len(expected)),
                 f"source row {i} left-pad/EOS/pad content or mask changed")
        if len(raw[i]["token_ids"]) <= offset:
            ending = raw[i]["token_ids"]
            _require(tail[:len(ending)] == ending and
                     all(v == pad for v in tail[len(ending):]) and
                     151645 in ending and ending.index(151645) == len(ending)-1,
                     f"ended companion {i} EOS/pad semantics changed")
    _require(ids[target, -SUFFIX:].tolist() == raw[target]["token_ids"][g["history_raw_offset"]:offset] and
             ids[target, -1].item() == case["native_x1_token"] and
             raw[target]["token_ids"][offset] == case["native_y1_token"] and
             g["physical_full_batch_padded"]["S"] == [width-SUFFIX, width] and
             pos[:, target, -SUFFIX:].cpu().tolist() == g["rotary_position_ids"]["S"],
             "target native first-y1 prefix changed")
    for name in ("earlier", "latest"):
        a, b = g["physical_full_batch_padded"][name]
        _require(b-a == 9 and pos[:, target, a:b].cpu().tolist() == g["rotary_position_ids"][name],
                 f"{name} physical/rotary row changed")
    _require(all(mask[i, -SUFFIX:].tolist() == [1] * SUFFIX for i in range(4)),
             "source companion suffix masks changed")


def verify_cross(native, cross, case):
    target = case["batch_index"]
    _require(target in range(4) and native["input_ids"].shape[0] == 4 and
             torch.equal(native["input_ids"], cross["input_ids"]) and
             torch.equal(native["attention_mask"], cross["attention_mask"]) and
             torch.equal(native["position_ids"][:, :, :-SUFFIX], cross["position_ids"][:, :, :-SUFFIX]) and
             torch.equal(native["position_ids"][:, [i for i in range(4) if i != target], -SUFFIX:],
                         cross["position_ids"][:, [i for i in range(4) if i != target], -SUFFIX:]) and
             torch.equal(cross["position_ids"][:, target, -SUFFIX:],
                         native["position_ids"][:, target, -SUFFIX:] - 9),
             "cross changed wrong target, content, companion, or phase sign")


def cpu_mutations(base, case, raw, prompts, pad):
    target = case["batch_index"]
    verify_full(base, case, raw, prompts, pad)
    donor = base["position_ids"][:, target, -SUFFIX:] - 9
    crossed = swap_prefix_positions(base, donor, target, SUFFIX,
                                    tuple(base["input_ids"][target, -SUFFIX:].tolist()))
    verify_cross(base, crossed, case)
    checks = []
    def must_reject(label, call):
        try: call()
        except (ValueError, AssertionError): checks.append(label)
        else: raise AssertionError(f"CPU verifier missed {label}")
    wrong = dict(case); wrong["batch_index"] = (target+1) % 4
    must_reject("wrong_target_index", lambda: verify_cross(base, crossed, wrong))
    for label, field, index, delta in (
        ("wrong_S_sign", "position_ids", (slice(None), target, slice(-SUFFIX, None)), 9),
        ("wrong_S_position", "position_ids", (slice(None), target, -1), 1),
        ("companion_content", "input_ids", ((target+1)%4, -1), 1),
        ("companion_position", "position_ids", (slice(None), (target+1)%4, -1), 1),
    ):
        edited = dict(crossed); edited[field] = crossed[field].clone(); edited[field][index] += delta
        must_reject(label, lambda e=edited: verify_cross(base, e, case))
    bad = dict(case); bad["geometry"] = dict(case["geometry"])
    bad["geometry"]["physical_full_batch_padded"] = dict(case["geometry"]["physical_full_batch_padded"])
    a, b = bad["geometry"]["physical_full_batch_padded"]["earlier"]
    bad["geometry"]["physical_full_batch_padded"]["earlier"] = [a+1, b+1]
    must_reject("wrong_historical_row_span", lambda: verify_full(base, bad, raw, prompts, pad))
    ended = [i for i in range(4) if len(raw[i]["token_ids"]) <= case["geometry"]["first_y1_raw_offset"]]
    for i in ended:
        edited = dict(base); edited["input_ids"] = base["input_ids"].clone()
        edited["input_ids"][i, -1] = case["native_x1_token"]
        must_reject(f"ended_companion_{i}_pad_mutation",
                    lambda e=edited: verify_full(e, case, raw, prompts, pad))
    return checks


def cost_forecast(m):
    first = m["selected"][0]
    r1 = json.loads(bound(json.loads(ADMISSION.read_text())["first_case_acceptance"]).read_text())
    # The accepted R1 fresh run is seven model and two vision calls.
    pilot = json.loads(bound(r1["cold_readback"]["pilot"]).read_text())
    seconds = float(pilot["cost"]["allocated_gpu_seconds"])
    _require(pilot["cost"]["model_forwards"] == 7 and pilot["cost"]["vision_forwards"] == 2 and
             abs(seconds + 14.241613768041134 - PRIOR_SECONDS) < 1e-6,
             "accepted R1 cost basis changed")
    values = []
    for case in m["selected"][1:]:
        pixel_ratio = case["pixel_elements_full_batch"] / first["pixel_elements_full_batch"]
        width_ratio = case["geometry"]["source_step_full_batch_width"] / first["geometry"]["source_step_full_batch_width"]
        seconds_2x = seconds * pixel_ratio * width_ratio * 2
        values.append({"case_id":case["id"], "pixel_ratio":pixel_ratio,
                       "width_ratio":width_ratio, "forecast_seconds_2x":seconds_2x})
    total = sum(v["forecast_seconds_2x"] for v in values)
    admission = json.loads(ADMISSION.read_text())
    _require(total < admission["budget"]["remaining_package_gpu_seconds"] and
             admission["budget"]["sequence_cumulative_prior_to_remaining"] + total/3600 < 8,
             "seven-case shape forecast exceeds admission")
    return {"r1_fresh_seconds":seconds,"r1_pilot":r1["cold_readback"]["pilot"],
            "planning_margin":2,"per_case":values,"remaining_seconds_2x":total,
            "charged_prior_seconds":PRIOR_SECONDS,"package_cap_seconds":3600,
            "sequence_prior_hours":admission["budget"]["sequence_cumulative_prior_to_remaining"]}


def preflight(out):
    _require(not out.exists(), "remaining-case output already exists")
    m, _, _, admission = contract(CASE_IDS[0])
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    _require(q.model is None, "CPU preflight loaded model")
    forecast = cost_forecast(m)
    records = []
    for case_id in CASE_IDS:
        _, case, receipt, _ = contract(case_id)
        raw = json.loads(Path(case["source_bindings"]["raw"]["path"]).read_text())["rows"]
        panel = json.loads((Path(case["source_bindings"]["raw"]["path"]).parents[3]/"panel.json").read_text())
        batch, raw, trace, group, planning = _source(source_boundary(case,raw),"untied",panel,q,torch.device("cpu"))
        _require(len(raw) == len(group["cases"]) == 4 and
                 input_identity(batch) == receipt["input_identity"] and
                 list(batch.request_ids) == case["source_bindings"]["input_identity_summary"]["request_ids"],
                 "CPU original four-request identity changed")
        pad = int(q.tokenizer.pad_token_id)
        full = source_inputs(_ConfigOnlyRope(),batch,raw,case,pad)
        checks = cpu_mutations(full,case,raw,batch.prompt_token_ids,pad)
        phase = phase_cpu_gate(case)
        offset = case["geometry"]["first_y1_raw_offset"]
        tails = _prefix_tokens(raw,offset,pad)
        endings = [{"batch_index":i,"request_id":batch.request_ids[i],"raw_length":len(raw[i]["token_ids"]),
                    "ended_before_source_step":len(raw[i]["token_ids"])<=offset,
                    "tail_last_five":tail[-5:],"suffix_input_last_five":full["input_ids"][i,-5:].tolist(),
                    "suffix_mask_last_five":full["attention_mask"][i,-5:].tolist()}
                   for i,tail in enumerate(tails)]
        records.append({"case_id":case_id,"source":case["source_bindings"],
                        "target_index":case["batch_index"],"input_identity_sha256":digest(input_identity(batch)),
                        "effective_identity_expected":receipt["identity"],
                        "full_source_input_hashes":{k:tensor_hash(full[k]) for k in
                            ("input_ids","attention_mask","position_ids","cache_position")},
                        "full_source_shape":{"batch_size":4,"target_index":case["batch_index"],
                            "width":int(full["input_ids"].shape[1]),
                            "image_grids":[list(g) for g in batch.image_grids],
                            "pixel_elements":int(batch.inputs["pixel_values"].numel())},
                        "source_step_offset":offset,"row_crosswalk":endings,
                        "cpu_checks":checks,"cpu_destination_phase":phase,
                        "source_planning_replanned":planning.get("replanned_image_plan")})
    out.mkdir(parents=True)
    sources = [REPO/x for x in IMPORTS]
    sources += [Path(inspect.getfile(x)) for x in (cache_utils,modeling_qwen3_vl,sdpa_attention)]
    captures = []
    for src in sources:
        rel = src.relative_to(REPO) if src.is_relative_to(REPO) else Path("transformers")/src.name
        saved = preserve_source(src,run_root=out,relative_name=rel)
        captures.append({"maintained":literal_binding(src),"capture":literal_binding(saved)})
    commands = []
    for index, case_id in enumerate(CASE_IDS):
        dest = out / f"{index+2:02d}-{case_id.split(':')[1]}-{case_id.split(':')[2]}"
        prefix = ["python","-B","-m","probes.recurrence_dynamics.recurrence_cross_image_phase.scale"]
        commands.append({"case_id":case_id,"output":str(dest),
            "gpu":prefix+["run","--output",str(dest),"--case",case_id,"--device","cuda:0"],
            "readback":prefix+["readback","--output",str(dest),"--case",case_id],
            "reduce":prefix+["reduce","--output",str(dest),"--case",case_id]})
    packet = {"schema":"recurrence_cross_image_phase.remaining_preflight.v1",
              "status":"cpu_qualified_before_gpu","protocol":literal_binding(PROTOCOL),
              "manifest":literal_binding(MANIFEST),"admission":literal_binding(ADMISSION),
              "r1_producer":admission["frozen_r1_producer"],
              "r1_candidate":admission["first_case_candidate"],
              "case_records":records,"cost_forecast":forecast,
              "direct_source_captures":captures,"commands":commands}
    _write_new(out/"preflight.json",packet)
    print(json.dumps({"status":packet["status"],"cases":len(records),
                      "ended_companions":sum(x["ended_before_source_step"] for r in records for x in r["row_crosswalk"]),
                      "forecast_seconds_2x":forecast["remaining_seconds_2x"],"captures":len(captures)}))

@contextmanager
def patched(cache,kind,keys,digest_before,spans,width,target):
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
                    old=layer.keys[target,:,a:b,:].clone()
                    saved.append((layer,old))
                    layer.keys[target,:,a:b,:].copy_(key)
            yield
        finally:
            cache.crop(width)
            if saved:
                a,b=spans["latest" if kind=="sham" else kind]
                for layer,old in saved:
                    layer.keys[target,:,a:b,:].copy_(old)
            _require(cache.get_seq_length()==width and cache_digest(cache)==digest_before,
                     "historical cache failed finally restoration")


def run(out,device_name,case_id):
    started=time.monotonic()
    m,case,source_receipt,admission=contract(case_id)
    target=case["batch_index"]
    root=out.parent
    completed=[json.loads(p.read_text()) for p in sorted(root.glob("*/receipt.json"))]
    _require(all(x["status"]=="candidate_complete" for x in completed) and
             len(completed)==CASE_IDS.index(case_id),"queue order or prior technical failure")
    cap_seconds=3600-PRIOR_SECONDS-sum(x["cost"]["allocated_gpu_seconds"] for x in completed)
    pre=json.loads((root/"preflight.json").read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and pre["manifest"]==literal_binding(MANIFEST) and
             pre["admission"]==literal_binding(ADMISSION) and
             pre["r1_producer"]==admission["frozen_r1_producer"] and
             pre["r1_candidate"]==admission["first_case_candidate"] and
             any(x["case_id"]==case_id for x in pre["case_records"]) and
             sum(x["forecast_seconds_2x"] for x in pre["cost_forecast"]["per_case"][len(completed):])<cap_seconds and
             device_name=="cuda:0" and not (out/"receipt.json").exists(),
             "GPU remaining-case authority changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]);bound(item["capture"])
    planned=next(x for x in pre["case_records"] if x["case_id"]==case_id)
    _require(cap_seconds>0 and not out.exists(),"package budget or case output exhausted")
    out.mkdir(parents=True)
    launch=_write_new(out/"launch.json",{"status":"frozen_before_model_load","pid":os.getpid(),
              "started_unix":time.time(),"preflight":literal_binding(root/"preflight.json"),
              "producer":literal_binding(Path(__file__)),"hard_cap_seconds":cap_seconds})
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
                 digest(input_identity(batch))==planned["input_identity_sha256"],
                 "GPU full source batch identity changed")
        full=source_inputs(model,batch,raw,case,int(q.tokenizer.pad_token_id))
        _require({k:tensor_hash(full[k]) for k in planned["full_source_input_hashes"]}==
                 planned["full_source_input_hashes"],"GPU full source input/positions changed")
        width=full["input_ids"].shape[1]-SUFFIX;end=width+SUFFIX
        spans={k:tuple(case["geometry"]["physical_full_batch_padded"][k])
               for k in ("earlier","latest")}
        cross_positions=full["position_ids"][:,target,width:end]-9
        cross=swap_prefix_positions(full,cross_positions,target,SUFFIX,
                                    tuple(full["input_ids"][target,width:end].tolist()))
        verify_cross(full,cross,case)
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
                     time.monotonic()-started<cap_seconds,"unexpected or over-cap model forward")
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
        trace_parity=[]
        offset=case["geometry"]["first_y1_raw_offset"]
        for i in range(4):
            if offset < len(raw[i]["token_ids"]):
                item=_trace_compare(logits=source_logits[i],trace=trace,batch_index=i,
                    absolute_offset=offset,token_id=int(raw[i]["token_ids"][offset]),
                    role="first_y1" if i==target else "original_companion_step",atol=TOL)
                _require(item["passed"],f"active source row {i} trace parity failed")
                trace_parity.append({"batch_index":i,"status":"active_trace_parity",**item})
            else:
                tail=_prefix_tokens(raw,offset,int(q.tokenizer.pad_token_id))[i]
                trace_parity.append({"batch_index":i,"status":"ended_before_source_step",
                    "raw_length":len(raw[i]["token_ids"]),"source_step_offset":offset,
                    "eos_token_id":151645,"pad_token_id":int(q.tokenizer.pad_token_id),
                    "suffix_last_five":tail[-5:],"trace_comparison":"undefined_after_end"})
        _require(trace_parity[target]["status"]=="active_trace_parity",
                 "target source trace undefined")
        full_dir=out/"cells/01-full-native";full_dir.mkdir(parents=True)
        torch.save(source_logits[target].clone(),full_dir/"vocabulary.pt")
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
            seen_phase.append({name:{"cos":cos[target,a:b].detach().clone(),
                                     "sin":sin[target,a:b].detach().clone()}
                               for name,(a,b) in spans.items()})
        capture.append(rotary.register_forward_hook(on_prefill_rotary))
        for index,layer in enumerate(model.model.language_model.layers):
            def on_pre_k(_module,_args,output,i=index):
                _require(output.shape==(4,width,HEADS,DIM),"normalized pre-K shape changed")
                pre_norm[i]={name:output[target,a:b].transpose(0,1).detach().clone()
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
            "earlier":tensor_hash(layer.keys[target,:,spans["earlier"][0]:spans["earlier"][1],:]),
            "latest":tensor_hash(layer.keys[target,:,spans["latest"][0]:spans["latest"][1],:]),
            "target_V":tensor_hash(layer.values[target]),
            "other_K":tensor_hash(layer.keys[[i for i in range(4) if i!=target]]),
            "other_V":tensor_hash(layer.values[[i for i in range(4) if i!=target]])}
            for layer in cache.layers]
        native_positions={name:full["position_ids"][:,target,a:b].cpu()
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
                "native_k":layer.keys[target,:,a:b,:].detach().float().cpu().clone(),
                "native_v":layer.values[target,:,a:b,:].detach().float().cpu().clone()}
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
            with patched(cache,kind,edits,native_digest,spans,width,target):
                expected=cache_digest(cache)
                for i,layer in enumerate(cache.layers):
                    _require(tensor_hash(layer.values[target])==native_segments[i]["target_V"] and
                             tensor_hash(layer.keys[[j for j in range(4) if j!=target]])==native_segments[i]["other_K"] and
                             tensor_hash(layer.values[[j for j in range(4) if j!=target]])==native_segments[i]["other_V"],
                             "target V or companion historical K/V changed")
                    for row in ("earlier","latest"):
                        a,b=spans[row]
                        intended=(kind==row or kind=="sham" and row=="latest")
                        required=tensor_hash(edits[i]) if intended else native_segments[i][row]
                        _require(tensor_hash(layer.keys[target,:,a:b,:])==required,
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
                                        "selected_K_sha256":{name:tensor_hash(current.keys[target,:,a:b,:])
                                                             for name,(a,b) in spans.items()},
                                        "target_V_sha256":tensor_hash(current.values[target]),
                                        "companion_K_sha256":tensor_hash(current.keys[[j for j in range(4) if j!=target]]),
                                        "companion_V_sha256":tensor_hash(current.values[[j for j in range(4) if j!=target]])}
                    def before_output(_module,_args,i=index):
                        _require(cache.get_seq_length(i)==end,"S keys not appended before attention output")
                        current=cache.layers[i]
                        _require({axis:tensor_hash(getattr(current,axis)[:,:,:width,:])
                                  for axis in ("keys","values")}==expected[i],
                                 "historical cache mutated during suffix")
                        after_rows[i]={"companion_suffix_K_sha256":
                                       tensor_hash(current.keys[[j for j in range(4) if j!=target],:,width:end,:]),
                                       "companion_suffix_V_sha256":
                                       tensor_hash(current.values[[j for j in range(4) if j!=target],:,width:end,:]),
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
                    err=max(float((rotary_values[0][j][target].detach().cpu()-
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
                vec=logits[target].clone();torch.save(vec,cell_dir/"vocabulary.pt")
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
                                              for i in range(4) if i!=target],
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
            _require(time.monotonic()-started<cap_seconds,"remaining package cap including setup reached")
        _require(counts=={"model_forwards":7,"vision_forwards":2} and len(done)==6,
                 "finite seven-call case incomplete")
        cost={"allocated_gpu_seconds":time.monotonic()-started,
              "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
              "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),**counts}
        _require(cost["allocated_gpu_seconds"]<cap_seconds and
                 PRIOR_SECONDS+sum(x["cost"]["allocated_gpu_seconds"] for x in completed)+cost["allocated_gpu_seconds"]<3600 and
                 admission["budget"]["sequence_cumulative_prior_to_remaining"]+
                 (sum(x["cost"]["allocated_gpu_seconds"] for x in completed)+cost["allocated_gpu_seconds"])/3600<8,
                 "package/sequence GPU cap exceeded")
        pilot=_write_new(out/"pilot.json",{
            "schema":"recurrence_cross_image_phase.pilot.v1","status":"candidate_complete",
            "protocol":literal_binding(PROTOCOL),"manifest":literal_binding(MANIFEST),
            "admission":literal_binding(ADMISSION),"r1_acceptance":admission["first_case_acceptance"],
            "preflight":literal_binding(root/"preflight.json"),"launch":launch,
            "effective_identity":identity,"input_identity_sha256":digest(input_identity(batch)),
            "source_planning_replanned":planning.get("replanned_image_plan"),
            "full_native":done[0]["record"],"prefill":literal_binding(out/"prefill-raw.json"),
            "destination_phase":literal_binding(out/"destination-phase-raw.json"),
            "destination_s_check":literal_binding(out/"destination-s-check.json"),
            "phase_qualification":literal_binding(out/"phase-qualification.json"),
            "completed":done,"cost":cost,
            "case_id":case_id,"target_index":target})
        terminal={"status":"candidate_complete","terminal":True,"pilot":pilot,
                  "completed_cells":len(done),"cost":cost,
                  "case_id":case_id}
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


def cold(out,case_id):
    m,case,receipt,admission=contract(case_id)
    target=case["batch_index"];root=out.parent
    pre=json.loads((root/"preflight.json").read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and
             pre["manifest"]==literal_binding(MANIFEST) and
             pre["admission"]==literal_binding(ADMISSION),"cold preflight changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]);bound(item["capture"])
    planned=next(x for x in pre["case_records"] if x["case_id"]==case_id)
    term=json.loads((out/"receipt.json").read_text())
    _require(term["status"]=="candidate_complete" and term["terminal"] and
             term["case_id"]==case_id and term["completed_cells"]==6 and
             term["cost"]["model_forwards"]==7 and term["cost"]["vision_forwards"]==2,
             "cold terminal calls incomplete")
    pilot=json.loads(bound(term["pilot"]).read_text())
    identity,saved=pilot["effective_identity"],receipt["identity"]
    _require(pilot["preflight"]==literal_binding(root/"preflight.json") and
             pilot["admission"]==literal_binding(ADMISSION) and
             pilot["r1_acceptance"]==admission["first_case_acceptance"] and
             pilot["case_id"]==case_id and pilot["target_index"]==target and
             {k:v for k,v in identity.items() if k!="loader_source"}==
             {k:v for k,v in saved.items() if k!="loader_source"} and
             all(identity["loader_source"][k]==saved["loader_source"][k]
                 for k in ("sha256","size_bytes")) and
             pilot["input_identity_sha256"]==planned["input_identity_sha256"] and
             [x["kind"] for x in pilot["completed"]]==
             ["full_native","native","cross","sham","latest","earlier"],
             "cold provenance/call order changed")
    full=json.loads(bound(pilot["full_native"]).read_text())
    consumer=json.loads(bound(full["consumer"]).read_text())
    full_vectors=torch.load(bound(full["full_batch_vector"]),map_location="cpu",weights_only=True)
    _require(full_vectors.ndim==2 and full_vectors.shape[0]==4 and
             torch.equal(full_vectors[target],torch.load(bound(full["vector"]),map_location="cpu",weights_only=True)) and
             all(x["passed"] for x in full["trace_parity"] if x["status"]=="active_trace_parity") and
             len(full["trace_parity"])==4 and consumer["media_present"] and
             not consumer["use_cache"] and not consumer["past_cache_present"],
             "cold full native source vector/consumer changed")
    raw=json.loads(Path(case["source_bindings"]["raw"]["path"]).read_text())["rows"]
    offset=case["geometry"]["first_y1_raw_offset"]
    prompts=receipt["input_identity"]["prompt_token_ids"]
    pad=int(AutoTokenizer.from_pretrained(BASE,local_files_only=True).pad_token_id)
    tails=_prefix_tokens(raw,offset,pad)
    _require(all((x["status"]=="active_trace_parity") == (offset<len(raw[i]["token_ids"]))
                 for i,x in enumerate(full["trace_parity"])) and
             all(x["batch_index"]==i for i,x in enumerate(full["trace_parity"])),
             "cold active/ended source trace semantics changed")
    histories=[list(p)+tail for p,tail in zip(prompts,tails,strict=True)]
    from src.qwen.native import padded_histories
    ids,mask=padded_histories(histories,pad_token_id=pad)
    _require(all(x["suffix_last_five"]==tails[i][-5:] for i,x in enumerate(full["trace_parity"])
                 if x["status"]=="ended_before_source_step"),
             "cold ended companion EOS/pad tail changed")
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
             planned["full_source_input_hashes"],
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
             torch.equal(native_full[target],vectors["native"]) and
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
        if kind=="cross":expected_pos[:,target]-=9
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



def reduce(out,case_id):
    m,case,_,_=contract(case_id)
    readback=cold(out,case_id)
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
    summary={"schema":"recurrence_cross_image_phase.remaining_case_reduction.v1",
             "status":"candidate","case_id":case_id,"category":category,
             "decision":m["decision"],"baseline_tv":baseline,"treatments":treatments,
             "earlier_minus_latest_ratio":difference,
             "cells":{kind:{"winner":cell["top2_ids"][0] if "top2_ids" in cell else int(logits[kind].argmax()),
                            "runner":cell["top2_ids"][1] if "top2_ids" in cell else int(torch.topk(logits[kind],2).indices[1]),
                            "gap":cell["top2_gap"] if "top2_gap" in cell else float(torch.topk(logits[kind],2).values.diff().abs()[0]),
                            "native_y1_log_probability":float(torch.log_softmax(logits[kind],-1)[case["native_y1_token"]]),
                            "vector":cell["vector"]}
                      for kind,cell in cells.items()},
             "cold_readback":readback,"cost":pilot["cost"],
             "admission":literal_binding(ADMISSION)}
    _write_new(out/"reduction.json",summary)
    print(json.dumps({"category":category,"baseline_tv":baseline,
                      "latest_ratio":latest,"earlier_ratio":earlier,
                      "case_gpu_seconds":pilot["cost"]["allocated_gpu_seconds"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--case",choices=CASE_IDS)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="preflight":preflight(args.output)
    else:
        _require(args.case is not None,"case ID required")
        if args.mode=="run":run(args.output,args.device,args.case)
        elif args.mode=="readback":print(json.dumps(cold(args.output,args.case)))
        else:reduce(args.output,args.case)


if __name__=="__main__":main()
