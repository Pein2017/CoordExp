"""Frozen two-case adapter for the accepted first-case K/V consumer path."""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from probes.recurrence_dynamics.recurrence_cross_image_phase import scale as cross
from probes.recurrence_dynamics.recurrence_y1_value_phase import run as first

REPO=Path(__file__).resolve().parents[3]
UNIT=REPO/"research/experiments/2026-09-23-recurrence-y1-value-phase"
ADMISSION=UNIT/"lead-remaining-admission-v1.json"
ADMISSION_SHA="60f71766264edf50de46fd464ff99e273b8445f8ed01d1bc1b9183c6bbc8a710"
ACCEPTANCE=UNIT/"lead-first-case-acceptance.json"
ACCEPTANCE_SHA="ef2a64d4adc0f51fb1486338d627a6a44fd638a99adad40b71515005bf1196aa"
FIRST_SHA="6e606141af0efae208151b07c1f662937743891d2be7f52813da1b32860a309e"
CASES=("mature:2299:2","mature:4134:7")
ROOT=Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/remaining-v1")
FIRST_SECONDS=56.56270640343428


def require(ok,message):
    first.require(bool(ok),message)


def bound(path):
    return first.binding(path)


def authority():
    require(bound(ADMISSION)["sha256"]==ADMISSION_SHA and
            bound(ACCEPTANCE)["sha256"]==ACCEPTANCE_SHA and
            bound(Path(first.__file__))["sha256"]==FIRST_SHA and
            bound(first.PROTOCOL)["sha256"]==first.PROTOCOL_SHA and
            bound(first.MANIFEST)["sha256"]==first.MANIFEST_SHA,
            "remaining admission or accepted first-case source changed")
    a=json.loads(ADMISSION.read_text())
    require(a["status"]=="lead-admitted-remaining-two" and
            a["case_ids"]==list(CASES) and
            a["budget"]["prior_package_gpu_seconds"]==FIRST_SECONDS and
            a["budget"]["additional_model_forwards_cap"]==14 and
            a["budget"]["additional_vision_forwards_cap"]==4 and
            a["budget"]["package_total_model_forwards_cap"]==21 and
            a["budget"]["package_total_vision_forwards_cap"]==6 and
            a["budget"]["package_gpu_hours_cap"]==.25 and
            a["budget"]["sequence_gpu_hours_cap"]==8 and
            a["budget"]["sequence_prior_gpu_hours"]==.20539820235884854 and
            a["budget"]["measured_shape_reforecast_seconds_2x"]==230.30549990789007,
            "remaining order, cost, or finite call contract changed")
    for key in ("protocol","manifest","first_case_acceptance","first_case_producer"):
        cross.bound(a[key])
    accepted=json.loads(ACCEPTANCE.read_text())
    for key in ("candidate","reduction","receipt"):
        cross.bound(accepted[key])
    require(accepted["status"]=="lead-accepted-first-case-only" and
            accepted["package_gpu_seconds"]==FIRST_SECONDS and
            accepted["sequence_gpu_hours"]==a["budget"]["sequence_prior_gpu_hours"] and
            accepted["checks"]["cold"]["status"]=="passed" and
            json.loads(Path(accepted["receipt"]["path"]).read_text())["cost"]["allocated_gpu_seconds"]==FIRST_SECONDS,
            "accepted first-case receipt or charge changed")
    return a


def case_gate(c,source,planned=None):
    g=source["geometry"];target=source["batch_index"]
    spans=g["physical_full_batch_padded"]
    require(c["id"]==source["id"] and target in (1,3) and
            c["V_row_offset"]==5 and
            c["donor_V_physical"]==spans["earlier"][0]+5 and
            c["destination_V_physical"]==spans["latest"][0]+5 and
            spans["earlier"][1]==spans["latest"][0] and
            spans["latest"][1]==spans["S"][0] and
            spans["S"][1]==g["source_step_full_batch_width"] and
            g["left_pad_target"]+g["physical_unpadded"]["latest"][0]==spans["latest"][0] and
            all(g["rotary_position_ids"]["S"][axis][4]-
                g["rotary_position_ids"]["latest"][axis][5]==8
                for axis in range(3)) and
            [c["roles"][name]["coordinate_token_id"] for name in ("earlier","latest","current")]==
            [c["roles"][name]["source_trace_chosen_token_id"] for name in ("earlier","latest","current")],
            "remaining target, pad, V slot, donor, phase, or source role changed")
    if planned is not None:
        require(planned["case_id"]==c["id"] and planned["target_index"]==target and
                planned["source_step_offset"]==g["first_y1_raw_offset"] and
                planned["full_source_shape"]["width"]==g["source_step_full_batch_width"] and
                planned["full_source_shape"]["batch_size"]==4 and
                planned["full_source_shape"]["pixel_elements"]==source["pixel_elements_full_batch"] and
                planned["ended_companions"]==[False]*4,
                "CPU full-batch caller plan changed")


def charge_gate(a,prior):
    require(prior>=FIRST_SECONDS and prior<900 and
            a["budget"]["prior_package_gpu_seconds"]==FIRST_SECONDS and
            900-prior>0 and
            a["budget"]["sequence_prior_gpu_hours"]+(prior-FIRST_SECONDS)/3600<8,
            "missing/changed prior charge or exhausted package")


def selected_contract(case_id):
    a=authority();require(case_id in CASES,"case outside remaining admission")
    m=json.loads(first.MANIFEST.read_text())
    c=m["cases"][1+CASES.index(case_id)]
    _,source,receipt,_=cross.contract(case_id)
    require(c["frozen_source_case"]==source,"frozen source case changed")
    for key in ("prefill_blocks","reference_reduction","reference_native_vector","reference_phase_vector"):
        cross.bound(c[key])
    case_gate(c,source)
    dynamic=copy.deepcopy(m)
    dynamic["held_case_ids"]=list(CASES[CASES.index(case_id)+1:])
    dynamic["admitted_case_ids"]=[case_id]
    previous=0.
    if case_id==CASES[1]:
        prior=ROOT/"01-2299-2/case/scale-receipt.json"
        require(prior.exists(),"D2 terminal charge missing before D4")
        previous=json.loads(prior.read_text())["allocated_gpu_seconds"]
    dynamic["budget"]["sequence_cumulative_prior_gpu_hours"]=a["budget"]["sequence_prior_gpu_hours"]+previous/3600
    return dynamic,c,source,receipt


@contextmanager
def bind_first_operations(case_id):
    saved=(first.CASE_ID,first.contract)
    try:
        first.CASE_ID=case_id
        first.contract=lambda: selected_contract(case_id)
        yield
    finally:
        first.CASE_ID,first.contract=saved


def mutation_gate(c,source,a):
    case_gate(c,source)
    checks=[]
    for name,change in (
        ("wrong_target",lambda x:x["frozen_source_case"].__setitem__("batch_index",(source["batch_index"]+1)%4)),
        ("wrong_slot",lambda x:x.__setitem__("destination_V_physical",x["destination_V_physical"]+1)),
        ("wrong_donor",lambda x:x.__setitem__("donor_V_physical",x["destination_V_physical"])),
        ("wrong_phase",lambda x:x["frozen_source_case"]["geometry"]["rotary_position_ids"]["S"][0].__setitem__(4,x["frozen_source_case"]["geometry"]["rotary_position_ids"]["S"][0][4]+1))):
        bad=copy.deepcopy(c);change(bad)
        try:case_gate(bad,bad["frozen_source_case"])
        except ValueError:checks.append(name)
        else:raise ValueError("actual caller gate missed "+name)
    try:charge_gate(a,FIRST_SECONDS-1)
    except ValueError:checks.append("missing_prior_charge")
    else:raise ValueError("actual caller gate missed missing charge")
    require(len(checks)==5,"actual caller mutations incomplete")
    return checks


def preflight(root):
    require(not root.exists(),"remaining-case root already exists")
    a=authority();m=json.loads(first.MANIFEST.read_text())
    frozen=[]
    for index,case_id in enumerate(CASES):
        _,c,source,_=selected_contract(case_id) if index==0 else _planned_contract(case_id)
        checks=mutation_gate(c,source,a)
        case_root=root/f"{index+1:02d}-{source['image_id']}-{source['current_row']}"
        with _bind_preflight(case_id):
            first.preflight(case_root)
        base=json.loads((case_root/"preflight.json").read_text())
        case_gate(c,source,base)
        require(base["cpu_patch_fixture"]["status"]=="passed" and
                base["cpu_phase"]["status"]=="passed" and
                {"wrong_target","wrong_slot","wrong_donor","unrelated_keys","unrelated_values"}.issubset(
                    base["cpu_patch_fixture"]["mutation_rejections"]),
                "actual patch/phase CPU qualification missing")
        frozen.append({"case_id":case_id,"source":source["source_bindings"],
                       "target_index":source["batch_index"],"left_pad":source["geometry"]["left_pad_target"],
                       "donor_slot":c["donor_V_physical"],"destination_slot":c["destination_V_physical"],
                       "rotary_position_ids":source["geometry"]["rotary_position_ids"],
                       "base_preflight":bound(case_root/"preflight.json"),"caller_mutations":checks,
                       "root":str(case_root),"output":str(case_root/"case")})
    old_cost=[json.loads(Path(m["cases"][i+1]["reference_reduction"]["path"]).read_text())["cost"]["allocated_gpu_seconds"]
              for i in range(2)]
    scale=a["budget"]["measured_shape_reforecast_seconds_2x"]/sum(old_cost)
    for row,seconds in zip(frozen,old_cost,strict=True):
        row["forecast_seconds_2x"]=seconds*scale
    require(sum(x["forecast_seconds_2x"] for x in frozen)==
            a["budget"]["measured_shape_reforecast_seconds_2x"] and
            FIRST_SECONDS+sum(x["forecast_seconds_2x"] for x in frozen)<900 and
            a["budget"]["sequence_prior_gpu_hours"]+
            sum(x["forecast_seconds_2x"] for x in frozen)/3600<8,
            "remaining shape-aware cost forecast exceeds cap")
    saved=cross.preserve_source(Path(__file__),run_root=root,
                                relative_name=Path('probes/recurrence_dynamics/recurrence_y1_value_phase/scale.py'))
    commands=[]
    for row in frozen:
        prefix=["python","-B","-m","probes.recurrence_dynamics.recurrence_y1_value_phase.scale"]
        commands.append({"case_id":row["case_id"],"gpu":prefix+["run","--root",str(root),
                        "--case",row["case_id"],"--device","cuda:0"],
                        "readback":prefix+["readback","--root",str(root),"--case",row["case_id"]],
                        "reduce":prefix+["reduce","--root",str(root),"--case",row["case_id"]]})
    packet={"schema":"recurrence_y1_value_phase.remaining_preflight.v1",
            "status":"cpu_qualified_before_gpu","admission":bound(ADMISSION),
            "acceptance":bound(ACCEPTANCE),"first_producer":bound(first.__file__),
            "scale_source":{"maintained":bound(__file__),"capture":bound(saved)},
            "cases":frozen,"commands":commands,
            "cost_forecast":{"remaining_seconds_2x":sum(x["forecast_seconds_2x"] for x in frozen),
                             "prior_package_seconds":FIRST_SECONDS,"package_cap_seconds":900,
                             "sequence_prior_gpu_hours":a["budget"]["sequence_prior_gpu_hours"],
                             "artifact_plan_bytes":a["budget"]["artifact_planning_bytes"]}}
    first.old._write_new(root/"scale-preflight.json",packet)
    print(json.dumps({"status":packet["status"],"case_ids":list(CASES),
                      "forecast_seconds_2x":packet["cost_forecast"]["remaining_seconds_2x"],
                      "caller_mutations":[x["caller_mutations"] for x in frozen]}))


@contextmanager
def _bind_preflight(case_id):
    """D4 CPU plan has no preceding GPU receipt yet; run uses strict binding."""
    saved=(first.CASE_ID,first.contract)
    try:
        first.CASE_ID=case_id
        first.contract=lambda:_planned_contract(case_id)
        yield
    finally:
        first.CASE_ID,first.contract=saved


def _planned_contract(case_id):
    a=authority();m=json.loads(first.MANIFEST.read_text())
    require(case_id in CASES,"case outside queue")
    c=m["cases"][1+CASES.index(case_id)]
    _,source,receipt,_=cross.contract(case_id)
    require(c["frozen_source_case"]==source,"planned source differs")
    case_gate(c,source)
    for key in ("prefill_blocks","reference_reduction","reference_native_vector","reference_phase_vector"):
        cross.bound(c[key])
    dynamic=copy.deepcopy(m);dynamic["held_case_ids"]=list(CASES[CASES.index(case_id)+1:])
    dynamic["admitted_case_ids"]=[case_id]
    dynamic["budget"]["sequence_cumulative_prior_gpu_hours"]=a["budget"]["sequence_prior_gpu_hours"]
    return dynamic,c,source,receipt


def queue_gate(root,case_id,mode):
    require(root==ROOT and case_id in CASES,"output root or case changed")
    a=authority();pre=json.loads((root/"scale-preflight.json").read_text())
    require(pre["status"]=="cpu_qualified_before_gpu" and
            pre["admission"]==bound(ADMISSION) and pre["acceptance"]==bound(ACCEPTANCE) and
            pre["first_producer"]==bound(first.__file__) and
            pre["scale_source"]["maintained"]==bound(__file__) and
            pre["scale_source"]["capture"]==bound(pre["scale_source"]["capture"]["path"]) and
            [x["case_id"] for x in pre["cases"]]==list(CASES) and
            pre["cost_forecast"]["prior_package_seconds"]==FIRST_SECONDS and
            pre["cost_forecast"]["remaining_seconds_2x"]==
            a["budget"]["measured_shape_reforecast_seconds_2x"],
            "remaining CPU/source/cost preflight changed")
    index=CASES.index(case_id);row=pre["cases"][index]
    expected=root/f"{index+1:02d}-{case_id.split(':')[1]}-{case_id.split(':')[2]}"
    require(row["root"]==str(expected) and row["output"]==str(expected/"case") and
            row["base_preflight"]==bound(expected/"preflight.json") and
            row["caller_mutations"]==["wrong_target","wrong_slot","wrong_donor","wrong_phase","missing_prior_charge"] and
            pre["commands"][index]["case_id"]==case_id,
            "remaining case order, caller, or source binding changed")
    base=json.loads((expected/"preflight.json").read_text())
    with _bind_preflight(case_id):
        _,c,source,_=_planned_contract(case_id)
    case_gate(c,source,base)
    for item in base["direct_source_captures"]:
        cross.bound(item["maintained"]);cross.bound(item["capture"])
    require(base["cpu_patch_fixture"]["status"]=="passed" and
            base["cpu_phase"]["status"]=="passed" and
            base["input_identity_sha256"]==source["source_input_identity_sha256"],
            "remaining base CPU case qualification changed")
    prior=FIRST_SECONDS
    if index:
        earlier=root/"01-2299-2/case"
        prior_receipt=json.loads((earlier/"scale-receipt.json").read_text())
        earlier_base=json.loads((earlier/"receipt.json").read_text())
        earlier_readback=json.loads((earlier/"scale-readback.json").read_text())
        earlier_reduction=json.loads((earlier/"scale-reduction.json").read_text())
        require(prior_receipt["status"]=="candidate_complete" and
                prior_receipt["case_id"]==CASES[0] and
                prior_receipt["base_terminal"]==bound(earlier/"receipt.json") and
                earlier_base["status"]=="candidate_complete" and
                earlier_base["cost"]["model_forwards"]==7 and
                earlier_base["cost"]["vision_forwards"]==2 and
                earlier_readback["status"]=="passed" and
                earlier_readback["base_terminal"]==bound(earlier/"receipt.json") and
                earlier_reduction["status"]=="candidate" and
                earlier_reduction["base_reduction"]==bound(earlier/"reduction.json"),
                "D2 terminal/cold/reduction missing or changed before D4")
        prior+=prior_receipt["allocated_gpu_seconds"]
    charge_gate(a,prior)
    if mode=="run":
        require(not (expected/"case").exists() and
                prior+sum(x["forecast_seconds_2x"] for x in pre["cases"][index:])<900 and
                a["budget"]["sequence_prior_gpu_hours"]+
                (prior-FIRST_SECONDS+sum(x["forecast_seconds_2x"] for x in pre["cases"][index:]))/3600<8,
                "case output exists or charge-adjusted forecast exceeds cap")
    return a,row,prior


def run(root,case_id,device_name):
    started=time.monotonic();out=root/f"{CASES.index(case_id)+1:02d}-{case_id.split(':')[1]}-{case_id.split(':')[2]}"/"case"
    require(not out.exists(),"remaining case output exists; no retry")
    try:
        a,row,prior=queue_gate(root,case_id,"run")
        pre=json.loads((root/"scale-preflight.json").read_text())
        require(pre["commands"][CASES.index(case_id)]["gpu"][-2:]==["--device",device_name],
                "frozen remaining GPU command changed")
        with bind_first_operations(case_id):
            first.run(out,device_name)
        base_receipt=json.loads((out/"receipt.json").read_text())
        require(base_receipt["status"]=="candidate_complete" and
                base_receipt["cost"]["model_forwards"]==7 and
                base_receipt["cost"]["vision_forwards"]==2,
                "base seven-call case incomplete")
        seconds=time.monotonic()-started
        require(prior+seconds<900 and
                a["budget"]["sequence_prior_gpu_hours"]+
                (prior-FIRST_SECONDS+seconds)/3600<8,
                "remaining package/sequence cap reached")
        terminal={"status":"candidate_complete","terminal":True,"case_id":case_id,
                  "allocated_gpu_seconds":seconds,"base_allocated_gpu_seconds":
                  base_receipt["cost"]["allocated_gpu_seconds"],
                  "base_terminal":bound(out/"receipt.json"),
                  "scale_preflight":bound(root/"scale-preflight.json"),
                  "prior_package_seconds":prior,"cumulative_package_seconds":prior+seconds,
                  "model_forwards":7,"vision_forwards":2}
    except BaseException as exc:
        out.mkdir(parents=True,exist_ok=True)
        terminal={"status":"technical_invalid","terminal":True,"case_id":case_id,
                  "allocated_gpu_seconds":time.monotonic()-started,"error":repr(exc),
                  "traceback":traceback.format_exc(),
                  "base_terminal":bound(out/"receipt.json") if (out/"receipt.json").exists() else None}
    first.old._write_new(out/"scale-receipt.json",terminal)
    if terminal["status"]!="candidate_complete":raise RuntimeError(terminal["error"])
    print(json.dumps({"status":terminal["status"],"case_id":case_id,
                      "charged_seconds":terminal["allocated_gpu_seconds"],
                      "cumulative_package_seconds":terminal["cumulative_package_seconds"]}))


def readback(root,case_id):
    _,row,_=queue_gate(root,case_id,"readback")
    out=Path(row["output"])
    with bind_first_operations(case_id):
        result=first.cold(out)
    record={"status":"passed","case_id":case_id,"base_cold":result,
            "base_terminal":bound(out/"receipt.json"),
            "scale_terminal":bound(out/"scale-receipt.json")}
    first.old._write_new(out/"scale-readback.json",record)
    print(json.dumps({"status":record["status"],"case_id":case_id,
                      "actual_consumer_layers":result["actual_consumer_layers"]}))


def reduce(root,case_id):
    _,row,_=queue_gate(root,case_id,"reduce")
    out=Path(row["output"])
    readback_record=json.loads((out/"scale-readback.json").read_text())
    require(readback_record["status"]=="passed" and
            readback_record["base_terminal"]==bound(out/"receipt.json"),
            "scale cold readback not qualified")
    with bind_first_operations(case_id):
        first.reduce(out)
    result=json.loads((out/"reduction.json").read_text())
    record={"status":"candidate","case_id":case_id,"category":result["category"],
            "base_reduction":bound(out/"reduction.json"),
            "scale_readback":bound(out/"scale-readback.json"),
            "scale_terminal":bound(out/"scale-receipt.json")}
    first.old._write_new(out/"scale-reduction.json",record)
    print(json.dumps({"status":record["status"],"case_id":case_id,
                      "category":record["category"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--case",choices=CASES)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    require(args.root==ROOT,"frozen remaining output root changed")
    if args.mode=="preflight":preflight(args.root)
    else:
        require(args.case is not None,"remaining case required")
        if args.mode=="run":run(args.root,args.case,args.device)
        elif args.mode=="readback":readback(args.root,args.case)
        else:reduce(args.root,args.case)


if __name__=="__main__":main()
