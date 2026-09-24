"""Eight finite readouts: four native diagonals, then four S-position crosses."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.recurrence_first_arrivals.prepare import _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from probes.training_set_completion.recurrence_next_history_prediction.run import (
    boundary, contract as prior_contract, make_input, suffix,
)
from probes.training_set_completion.recurrence_position_history import observed_hooks, swap_prefix_positions
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-next-step-cancellation"
PROTOCOL = UNIT / "unit.md"
MANIFEST = UNIT / "manifest.json"
PROTOCOL_SHA = "c9b0a93c8e8665f49f5e83d21d590bf47a2c2ce061ab60ae631e529e27da1629"
MANIFEST_SHA = "53864f21ffa06b7a44bc0f3ac4643b6f1bd5131c80502eace2208c7f6fdec468"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001")
CAP_SECONDS = 900
TOL = 2e-4
IMPORTS = [
    "probes/training_set_completion/recurrence_next_step_cancellation/run.py",
    "probes/training_set_completion/recurrence_next_history_prediction/run.py",
    "probes/training_set_completion/recurrence_chair_history_position/run.py",
    "probes/training_set_completion/artifacts.py",
    "probes/training_set_completion/coordinate_continuity/runtime.py",
    "probes/training_set_completion/native_row_choice/runtime.py",
    "probes/training_set_completion/numerical_feedback/select.py",
    "probes/training_set_completion/recurrence_first_arrivals/prepare.py",
    "probes/training_set_completion/recurrence_first_arrivals/stage1_case.py",
    "probes/training_set_completion/recurrence_position_history.py",
    "probes/training_set_completion/untied_shared.py",
    "src/artifacts/source_provenance.py", "src/qwen/input_identity.py",
    "src/qwen/native.py", "src/qwen/runtime_loading.py", "src/inference/bound_requests.py",
]


def bound(item):
    p = Path(item["path"])
    if not p.is_absolute():
        p = REPO / p
    actual = literal_binding(p)
    _require(actual["sha256"] == item["sha256"] and actual["size_bytes"] == item["size_bytes"],
             f"bound file changed: {p}")
    return p


def contract():
    _require(literal_binding(PROTOCOL)["sha256"] == PROTOCOL_SHA and
             literal_binding(MANIFEST)["sha256"] == MANIFEST_SHA, "lead contract changed")
    m = json.loads(MANIFEST.read_text())
    _require(m["status"] == "lead-frozen-before-crossed-calls" and
             m["source_group"] == "refined-04" and m["batch_index"] == 0 and
             m["request_id"] == "coco2017_train_000000477415" and
             m["histories"] == {"L": 45, "U": 54} and
             m["positions"] == {"L": list(range(432,437)), "U": list(range(441,446))} and
             m["budget"]["model_forwards"] == m["budget"]["vision_forwards"] == 8 and
             m["budget"]["free_tokens"] == 0 and m["budget"]["incremental_gpu_hours_cap"] == .25 and
             m["budget"]["sequence_gpu_hours_cap"] == 8 and
             m["decision"] == {"factor_tv_multiplier": 2, "absolute_tv_guard": .001,
                              "compensation_cosine_max": -.5,
                              "strict_shared_requires": "both paths on both probes"},
             "finite source/decision/cost contract changed")
    for key in ("predecessor_acceptance", "predecessor_manifest", "predecessor_reduction"):
        bound(m[key])
    for key in ("raw", "trace", "runtime_receipt", "image"):
        bound(m["source_bindings"][key])
    old, receipt, native, panel = prior_contract()
    _require(m["source_bindings"] == old["source_bindings"] and
             native[45:54] == native[36:45] and native[54:58] == native[45:49] and
             len(json.loads(Path(m["source_bindings"]["raw"]["path"]).read_text())["rows"]) == 1 and
             receipt["input_identity"]["request_ids"] == [m["request_id"]],
             "single-request L/U history changed")
    _require([(p["owner"], p["x1_token"]) for p in m["probes"]] ==
             [(1589003,152180),(1586761,152088)] and
             [(r["owner"],r["history"]) for r in m["references"]] ==
             [(1589003,"L"),(1586761,"L"),(1589003,"U"),(1586761,"U")],
             "probes/references changed")
    for r in m["references"]:
        bound(r["cell"]); bound(r["vector"])
        cell = json.loads(Path(r["cell"]["path"]).read_text())
        _require(cell["vector"] == r["vector"] and cell["owner"] == r["owner"] and
                 cell["row"] == (5 if r["history"] == "L" else 6),
                 "accepted reference cell differs")
    return m, receipt, native, panel


def verify_consumer(expected, actual):
    """The same verifier is used by CPU mutation checks and the model pre-hook."""
    observed = {}
    for key in ("input_ids", "attention_mask", "position_ids"):
        value = actual.get(key)
        _require(isinstance(value, torch.Tensor) and torch.equal(value, expected[key]),
                 f"actual consumer {key} changed")
        observed[key] = value.detach().cpu().tolist()
    for key in ("pixel_values", "image_grid_thw"):
        value = actual.get(key)
        _require(isinstance(value, torch.Tensor) and
                 tensor_hash(value) == tensor_hash(expected[key]),
                 f"actual consumer {key} changed")
        observed[key+"_sha256"] = tensor_hash(value)
    return observed


def cpu_mutation_checks():
    sample = {"input_ids": torch.tensor([[11,12,13,14,15,16,17]]),
              "attention_mask": torch.ones(1,7,dtype=torch.long),
              "position_ids": torch.arange(7).expand(3,1,7).clone(),
              "pixel_values": torch.ones(2,3), "image_grid_thw": torch.tensor([[1,2,3]])}
    _require(verify_consumer(sample,sample)["input_ids"] == sample["input_ids"].tolist(),
             "consumer verifier cannot accept identity")
    for key, index in (("input_ids",(0,-2)), ("input_ids",(0,0)),
                       ("input_ids",(0,-1)), ("position_ids",(0,0,-1))):
        bad = dict(sample); bad[key] = sample[key].clone(); bad[key][index] += 1
        try:
            verify_consumer(sample,bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"mutated {key} escaped actual consumer verifier")
    donor = torch.arange(8,13).expand(3,5).clone()
    crossed = swap_prefix_positions(sample, donor, 0, 5, (13,14,15,16,17))
    _require(torch.equal(crossed["position_ids"][:,:,0:2], sample["position_ids"][:,:,0:2]) and
             torch.equal(crossed["input_ids"], sample["input_ids"]),
             "S swap changed history/content")
    for wrong in ((14,15,16,17,18),(13,14,15,16)):
        try:
            swap_prefix_positions(sample, donor, 0, 5, wrong)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("wrong S slot escaped swap")
    return ["identity accepted", "wrong S slot rejected", "extra history content rejected",
            "replaced supplied x1 rejected", "wrong S position rejected", "history preserved by swap"]


def probabilities(vec):
    value = vec.to(torch.float64)
    _require(value.ndim == 1 and torch.isfinite(value).all(), "invalid vocabulary vector")
    return torch.log_softmax(value,-1).exp()


def preflight(out):
    m, receipt, native, panel = contract()
    _require(not out.exists(), "attempt already exists")
    checks = cpu_mutation_checks()
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    _require(q.model is None, "CPU preflight loaded model")
    batch, raw, trace, group, planning = _source(boundary(m,native), "untied", panel, q, torch.device("cpu"))
    _require(len(raw) == len(group["cases"]) == 1 and
             input_identity(batch) == receipt["input_identity"], "original source identity changed")
    prompt = list(batch.prompt_token_ids[0]); _require(len(prompt) == 1362, "prompt width changed")
    ref_checks = []
    vectors = {}
    for r in m["references"]:
        owner,h = r["owner"],r["history"]
        x1 = next(p["x1_token"] for p in m["probes"] if p["owner"] == owner)
        row = 5 if h == "L" else 6
        consumer_path = Path(r["cell"]["path"]).parent / "consumer-raw.json"
        consumer = json.loads(consumer_path.read_text())
        expected = prompt + suffix(native,row,x1)
        _require(consumer["input_ids"] == [expected] and
                 consumer["attention_mask"] == [[1]*len(expected)] and
                 [axis[0][-5:] for axis in consumer["position_ids"]] ==
                 [m["positions"][h]]*3,
                 "accepted reference actual consumer differs")
        vectors[owner,h] = torch.load(bound(r["vector"]),map_location="cpu",weights_only=True)
        ref_checks.append({"owner":owner,"history":h,"consumer":literal_binding(consumer_path),
                           "physical_S_range":[len(prompt)+(45 if h=="L" else 54),
                                               len(prompt)+(50 if h=="L" else 59)],
                           "rotary_S_positions":[m["positions"][h]]*3})
    for p in m["probes"]:
        owner=p["owner"]
        tv=.5*float(torch.sum(torch.abs(probabilities(vectors[owner,"U"])-probabilities(vectors[owner,"L"]))))
        _require(abs(tv-m["native_tv"][str(owner)]) < 1e-10,
                 "frozen native TV changed")
    out.mkdir(parents=True)
    captures=[]
    for name in IMPORTS:
        source=REPO/name
        capture=preserve_source(source,run_root=out,relative_name=Path(name))
        captures.append({"maintained":literal_binding(source),"capture":literal_binding(capture)})
    old=json.loads(bound(m["predecessor_reduction"]).read_text())
    old_seconds=old["cost"]["allocated_gpu_seconds"]
    shape_factor=max(1.,1421/1421)
    estimate=2*old_seconds*(8/5)*shape_factor
    _require(estimate < CAP_SECONDS and
             m["budget"]["sequence_cumulative_prior_gpu_hours"]+estimate/3600 < 8,
             "shape-aware cost forecast exceeds cap")
    packet={"schema":"recurrence_next_step_cancellation.preflight.v1",
            "status":"cpu_qualified_before_gpu","protocol":literal_binding(PROTOCOL),
            "manifest":literal_binding(MANIFEST),"source":m["source_bindings"],
            "input_identity":input_identity(batch),"source_planning":planning,
            "shape":{"batch_size":1,"prompt_width":1362,"L_width":1412,"U_width":1421,
                     "image_grids":[list(x) for x in batch.image_grids],
                     "pixel_elements":int(batch.inputs["pixel_values"].numel())},
            "crosswalk":ref_checks,"cpu_actual_verifier_mutations":checks,
            "decision":m["decision"],"native_tv":m["native_tv"],
            "thresholds":{str(p["owner"]):max(2*m["native_tv"][str(p["owner"])],.001)
                          for p in m["probes"]},
            "direct_source_captures":captures,
            "cost_forecast":{"predecessor_five_call_seconds":old_seconds,"new_calls":8,
                             "shape_factor":shape_factor,"planning_margin":2,
                             "estimated_seconds":estimate,"incremental_cap_seconds":CAP_SECONDS,
                             "sequence_prior_hours":m["budget"]["sequence_cumulative_prior_gpu_hours"]},
            "commands":{"gpu":["python","-B","-m","probes.training_set_completion.recurrence_next_step_cancellation.run","run","--output",str(out),"--device","cuda:0"],
                        "readback":["python","-B","-m","probes.training_set_completion.recurrence_next_step_cancellation.run","readback","--output",str(out)],
                        "reduce":["python","-B","-m","probes.training_set_completion.recurrence_next_step_cancellation.run","reduce","--output",str(out)]}}
    _write_new(out/"preflight.json",packet)
    print(json.dumps({"status":packet["status"],"thresholds":packet["thresholds"],
                      "forecast_seconds":estimate,"source_shape":packet["shape"]}))


def run(out,device_name):
    started=time.monotonic()
    m,source_receipt,native,panel=contract()
    pre_path=out/"preflight.json"; pre=json.loads(pre_path.read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu" and
             pre["protocol"]==literal_binding(PROTOCOL) and pre["manifest"]==literal_binding(MANIFEST) and
             pre["decision"]==m["decision"] and device_name=="cuda:0" and
             not (out/"receipt.json").exists(),"GPU launch authority changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]);bound(item["capture"])
    launch=_write_new(out/"launch.json",{"status":"frozen_before_model_load","pid":os.getpid(),
                       "started_unix":time.time(),"preflight":literal_binding(pre_path),
                       "producer":literal_binding(Path(__file__)),"hard_cap_seconds":CAP_SECONDS})
    device=torch.device(device_name);counts={"model_forwards":0,"vision_forwards":0};completed=[];handles=[];active=None
    try:
        torch.cuda.set_device(device);torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        q,identity=load_model("untied",device)
        saved=source_receipt["identity"]
        _require({k:v for k,v in identity.items() if k!="loader_source"}==
                 {k:v for k,v in saved.items() if k!="loader_source"} and
                 all(identity["loader_source"][k]==saved["loader_source"][k]
                     for k in ("sha256","size_bytes")),"model identity changed")
        model=q.model.eval()
        batch,raw,trace,group,planning=_source(boundary(m,native),"untied",panel,q,device)
        _require(len(raw)==len(group["cases"])==1 and
                 input_identity(batch)==source_receipt["input_identity"]==pre["input_identity"],
                 "GPU source identity changed")
        pad=int(q.tokenizer.pad_token_id)
        bases={}
        for p in m["probes"]:
            for h,row in (("L",5),("U",6)):
                inp=make_input(model,batch,native,row,p["x1_token"],pad)
                _require(inp["position_ids"][:,0,-5:].tolist()==[m["positions"][h]]*3 and
                         inp["input_ids"].shape[1]==(1412 if h=="L" else 1421),
                         "native diagonal input/position crosswalk changed")
                bases[p["owner"],h]=inp
        cells={}
        for p in m["probes"]:
            owner,x1=p["owner"],p["x1_token"]
            for h in ("L","U"):
                original=bases[owner,h]
                native_pos=original["position_ids"][:,0,-5:].clone()
                cells[owner,h,h]=original
                other="U" if h=="L" else "L"
                donor=bases[owner,other]["position_ids"][:,0,-5:].clone()
                crossed=swap_prefix_positions(original,donor,0,5,tuple(suffix(native,5 if h=="L" else 6,x1)[-5:]))
                _require(torch.equal(crossed["input_ids"],original["input_ids"]) and
                         torch.equal(crossed["attention_mask"],original["attention_mask"]) and
                         torch.equal(crossed["position_ids"][:,:, :-5],original["position_ids"][:,:, :-5]) and
                         not torch.equal(donor,native_pos),"cross changed history or failed S translation")
                cells[owner,h,other]=crossed
        for r in m["references"]:
            x1=next(p["x1_token"] for p in m["probes"] if p["owner"]==r["owner"])
            inp=bases[r["owner"],r["history"]]
            saved_consumer=json.loads((Path(r["cell"]["path"]).parent/"consumer-raw.json").read_text())
            _require(inp["input_ids"].cpu().tolist()==saved_consumer["input_ids"] and
                     inp["position_ids"].cpu().tolist()==saved_consumer["position_ids"] and
                     inp["input_ids"][0,-1].item()==x1,"accepted diagonal consumer differs")

        def before_model(_module,_args,kwargs):
            counts["model_forwards"]+=1
            _require(active is not None and counts["model_forwards"]<=8 and
                     time.monotonic()-started<CAP_SECONDS,"unexpected or over-cap forward")
            active["observed"].update(verify_consumer(active["input"],kwargs))
        def before_vision(_module,_args):
            counts["vision_forwards"]+=1
            _require(counts["vision_forwards"]<=8,"vision cap exceeded")
        handles.extend((model.register_forward_pre_hook(before_model,with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision)))
        plan=[(p["owner"],h,h) for p in m["probes"] for h in ("L","U")]
        plan += [(p["owner"],h,"U" if h=="L" else "L") for p in m["probes"] for h in ("L","U")]
        _require(len(plan)==8,"eight-cell plan changed")
        qualified=[]
        for index,(owner,h,pos) in enumerate(plan,1):
            if index>4:
                _require(len(qualified)==4 and all(q["passed"] for q in qualified),
                         "four diagonal qualifications incomplete")
            inp=cells[owner,h,pos];active={"input":inp,"observed":{}}
            seen,scoped=observed_hooks(model,inp["position_ids"],5,0)
            rotary=[]
            extra=model.model.language_model.rotary_emb.register_forward_pre_hook(
                lambda _module,args:rotary.append(args[1].detach().cpu().tolist()))
            try:
                with torch.inference_mode():
                    vec=model(**inp).logits[0,-1].detach().float().cpu()
            finally:
                extra.remove()
                for hook in scoped:hook.remove()
            torch.cuda.synchronize(device)
            _require(len(rotary)==seen["rotary"]==1 and
                     active["observed"]["position_ids"]==rotary[0] and
                     len(seen["masks"])==len(seen["caches"])==seen["embedding"]==2,
                     "actual rotary/mask/cache evidence incomplete")
            observed={**active["observed"],"rotary_position_ids":rotary[0],
                      "attention_mask_hashes":seen["masks"],"cache_position_hashes":seen["caches"],
                      "position_embedding_hooks":seen["embedding"]}
            active=None
            cell_dir=out/"cells"/f"{index:02d}-{owner}-{h}-{pos}"
            cell_dir.mkdir(parents=True,exist_ok=False)
            torch.save(vec,cell_dir/"vocabulary.pt")
            vector=literal_binding(cell_dir/"vocabulary.pt")
            consumer=_write_new(cell_dir/"consumer-raw.json",observed)
            _require(vec.ndim==1 and torch.isfinite(vec).all(),"invalid vocabulary vector")
            top=torch.topk(vec,2)
            record={"owner":owner,"history":h,"positions":pos,"diagonal":h==pos,
                    "vector":vector,"consumer":consumer,"top2_ids":top.indices.tolist(),
                    "top2_logits":top.values.tolist(),"top2_gap":float(top.values[0]-top.values[1]),
                    "logsumexp":float(torch.logsumexp(vec,-1)),"vocabulary_size":int(vec.numel()),
                    "counters_after":dict(counts),"allocated_gpu_seconds_after":time.monotonic()-started}
            if h==pos:
                ref=next(r for r in m["references"] if r["owner"]==owner and r["history"]==h)
                old=torch.load(bound(ref["vector"]),map_location="cpu",weights_only=True)
                error=float((vec-old).abs().max())
                record["reference_max_abs_error"]=error
                qualified.append({"owner":owner,"history":h,"passed":error<=TOL,
                                  "max_abs_error":error})
                _require(error<=TOL,"diagonal full-vector qualification failed")
            _write_new(cell_dir/"cell.json",record)
            completed.append({"owner":owner,"history":h,"positions":pos,
                              "record":literal_binding(cell_dir/"cell.json")})
            _write_new(out/f"checkpoint-{index:02d}.json",
                       {"completed":completed,"counts":dict(counts),"allocated_gpu_seconds":time.monotonic()-started})
            _require(time.monotonic()-started<CAP_SECONDS,"incremental cap reached")
        _require(counts=={"model_forwards":8,"vision_forwards":8},"eight calls incomplete")
        cost={"allocated_gpu_seconds":time.monotonic()-started,
              "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
              "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),**counts}
        _require(cost["allocated_gpu_seconds"]<CAP_SECONDS and
                 m["budget"]["sequence_cumulative_prior_gpu_hours"]+cost["allocated_gpu_seconds"]/3600<8,
                 "cumulative GPU cap reached")
        pilot=_write_new(out/"pilot.json",{"schema":"recurrence_next_step_cancellation.pilot.v1",
                  "status":"candidate_complete","preflight":literal_binding(pre_path),
                  "launch":launch,"protocol":literal_binding(PROTOCOL),"manifest":literal_binding(MANIFEST),
                  "effective_identity":identity,"input_identity":input_identity(batch),
                  "source_planning":planning,"completed":completed,"qualifications":qualified,"cost":cost})
        terminal={"status":"candidate_complete","terminal":True,"pilot":pilot,
                  "completed_cells":8,"cost":cost}
    except BaseException as exc:
        terminal={"status":"technical_invalid","terminal":True,"error":repr(exc),
                  "traceback":traceback.format_exc(),"completed":completed,
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


def readback(out):
    m,_,native,_=contract()
    pre=json.loads((out/"preflight.json").read_text())
    _require(pre["status"]=="cpu_qualified_before_gpu","cold preflight invalid")
    for item in pre["direct_source_captures"]:bound(item["capture"])
    terminal=json.loads((out/"receipt.json").read_text())
    _require(terminal["terminal"] and terminal["status"]=="candidate_complete" and
             terminal["completed_cells"]==8,"cold terminal invalid")
    pilot=json.loads(bound(terminal["pilot"]).read_text())
    _require(len(pilot["completed"])==8 and pilot["cost"]["model_forwards"]==
             pilot["cost"]["vision_forwards"]==8 and len(pilot["qualifications"])==4 and
             all(q["passed"] for q in pilot["qualifications"]),"cold qualification/count invalid")
    cells={}
    for item in pilot["completed"]:
        cell=json.loads(bound(item["record"]).read_text())
        vec=torch.load(bound(cell["vector"]),map_location="cpu",weights_only=True)
        observed=json.loads(bound(cell["consumer"]).read_text())
        _require(vec.ndim==1 and vec.numel()==cell["vocabulary_size"] and
                 torch.isfinite(vec).all() and int(vec.argmax())==cell["top2_ids"][0] and
                 math.isclose(float(torch.logsumexp(vec,-1)),cell["logsumexp"],abs_tol=1e-6) and
                 observed["position_ids"]==observed["rotary_position_ids"] and
                 len(observed["attention_mask_hashes"])==len(observed["cache_position_hashes"])==2,
                 "cold vector/consumer invalid")
        owner,h,pos=cell["owner"],cell["history"],cell["positions"]
        ref=next(r for r in m["references"] if r["owner"]==owner and r["history"]==h)
        accepted=json.loads((Path(ref["cell"]["path"]).parent/"consumer-raw.json").read_text())
        _require(observed["input_ids"]==accepted["input_ids"] and
                 observed["attention_mask"]==accepted["attention_mask"] and
                 observed["pixel_values_sha256"]==accepted["pixel_values_sha256"] and
                 observed["image_grid_thw_sha256"]==accepted["image_grid_thw_sha256"] and
                 [axis[0][:-5] for axis in observed["position_ids"]]==
                 [axis[0][:-5] for axis in accepted["position_ids"]] and
                 [axis[0][-5:] for axis in observed["position_ids"]]==[m["positions"][pos]]*3 and
                 len(observed["input_ids"][0])==(1412 if h=="L" else 1421),
                 "cold actual history/image/position crossing differs")
        if h==pos:
            _require(observed["position_ids"]==accepted["position_ids"],
                     "cold diagonal position differs from accepted native")
        cells[owner,h,pos]=cell
    expected={(p["owner"],h,pos) for p in m["probes"] for h in ("L","U") for pos in ("L","U")}
    _require(set(cells)==expected,"cold eight-cell grid differs")
    return {"status":"passed","pilot":literal_binding(out/"pilot.json"),
            "terminal":literal_binding(out/"receipt.json"),"cells":8,
            "allocated_gpu_seconds":pilot["cost"]["allocated_gpu_seconds"]}


def tv(a,b):
    return .5*float(torch.sum(torch.abs(a-b)))


def cosine(a,b):
    an=float(torch.linalg.vector_norm(a));bn=float(torch.linalg.vector_norm(b))
    return None if an==0 or bn==0 else float(torch.dot(a,b)/(an*bn))


def reduce(out):
    m,_,_,_=contract();cold=readback(out)
    pilot=json.loads((out/"pilot.json").read_text())
    cells={(z["owner"],z["history"],z["positions"]):
           json.loads(Path(z["record"]["path"]).read_text()) for z in pilot["completed"]}
    results={}
    for probe in m["probes"]:
        owner=probe["owner"]
        raw={(h,pos):torch.load(Path(cells[owner,h,pos]["vector"]["path"]),
                                map_location="cpu",weights_only=True).double()
             for h in ("L","U") for pos in ("L","U")}
        p={key:probabilities(v) for key,v in raw.items()}
        a,b,c,d=(p["L","L"],p["L","U"],p["U","L"],p["U","U"])
        net=d-a;native_tv=tv(d,a)
        _require(abs(native_tv-m["native_tv"][str(owner)])<1e-10,"native net TV changed")
        threshold=max(2*native_tv,.001)
        paths={}
        near=False
        for name,u,v in (("position_first",b-a,d-b),("history_first",c-a,d-c)):
            _require(float(torch.max(torch.abs(u+v-net)))<1e-12,"path decomposition identity failed")
            t1,t2=tv(u,torch.zeros_like(u)),tv(v,torch.zeros_like(v))
            cos=cosine(u,v)
            if abs(t1-threshold)<=1e-8 or abs(t2-threshold)<=1e-8 or (cos is not None and abs(cos+.5)<=1e-8):
                near=True
            paths[name]={"first_step_tv":t1,"second_step_tv":t2,"cosine":cos,
                         "path_length_over_native_net":None if native_tv==0 else (t1+t2)/native_tv,
                         "strict_compensation_path":t1>threshold and t2>threshold and cos is not None and cos<-.5}
        interaction=d-b-c+a
        steps=[v for x in paths.values() for k,v in x.items() if k in ("first_step_tv","second_step_tv")]
        uniform=all(v<=threshold for v in steps)
        compensate=all(x["strict_compensation_path"] for x in paths.values())
        category="numerical_HOLD" if near else "compensation" if compensate else "uniformly_small" if uniform else "mixed"
        pair=(151670+408,151670+413) if owner==1589003 else (151670+999,151670+0)
        results[str(owner)]={"native_tv":native_tv,"factor_threshold":threshold,
             "paths":paths,"interaction":{"l1":float(torch.linalg.vector_norm(interaction,ord=1)),
                                      "l2":float(torch.linalg.vector_norm(interaction,ord=2))},
             "category":category,"cells":{f"{h}/{pos}":{
                 "winner":cells[owner,h,pos]["top2_ids"][0],
                 "runner":cells[owner,h,pos]["top2_ids"][1],
                 "gap":cells[owner,h,pos]["top2_gap"],
                 "winning_pair_margin_logit":float(raw[h,pos][pair[0]]-raw[h,pos][pair[1]]),
                 "raw_vector":cells[owner,h,pos]["vector"],
                 "actual_consumer":cells[owner,h,pos]["consumer"]}
                 for h in ("L","U") for pos in ("L","U")}}
    shared=all(x["category"]=="compensation" for x in results.values())
    uniformly_small=all(x["category"]=="uniformly_small" for x in results.values())
    reduction={"schema":"recurrence_next_step_cancellation.reduction.v1",
               "status":"candidate_complete","protocol":literal_binding(PROTOCOL),
               "manifest":literal_binding(MANIFEST),"preflight":literal_binding(out/"preflight.json"),
               "pilot":literal_binding(out/"pilot.json"),"terminal":literal_binding(out/"receipt.json"),
               "cold_readback":cold,"results":results,
               "shared_strict_compensation":shared,"shared_uniformly_small":uniformly_small,
               "decision_constants":m["decision"],"cells":pilot["completed"],"cost":pilot["cost"],
               "cumulative_gpu_hours":m["budget"]["sequence_cumulative_prior_gpu_hours"]+
                                      pilot["cost"]["allocated_gpu_seconds"]/3600,
               "artifact_bytes_before_reduction":sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
    _write_new(out/"reduction.json",reduction)
    print(json.dumps({"status":reduction["status"],"shared_strict_compensation":shared,
                      "shared_uniformly_small":uniformly_small,
                      "categories":{k:v["category"] for k,v in results.items()},
                      "cost":pilot["cost"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="preflight":preflight(args.output)
    elif args.mode=="run":run(args.output,args.device)
    elif args.mode=="readback":
        result=readback(args.output);_write_new(args.output/"cold-readback.json",result);print(json.dumps(result))
    else:reduce(args.output)


if __name__=="__main__":main()
