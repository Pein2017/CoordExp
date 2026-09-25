"""Bound final D4 operation after the frozen R4 and D3 technical holds."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import FunctionType

from src.artifacts.utf8_json import literal_binding
from probes.recurrence_dynamics.recurrence_cross_image_phase import continuation, scale
from probes.recurrence_dynamics.recurrence_first_arrivals.prepare import _require
from probes.recurrence_dynamics.recurrence_first_arrivals.stage1_case import _write_new
from src.artifacts.source_provenance import preserve_source

UNIT = scale.UNIT
ADMISSION = UNIT / "lead-D4-admission-v1.json"
ADMISSION_SHA = "269eafd0b620a40f0ae7eecf53b6ef3cc3fdae9c31ee2a21e9cc2bc9d01f5247"
RULING = UNIT / "lead-D1-D2-acceptance-and-D3-ruling.json"
RULING_SHA = "f500a545b5012347a65fd6039b7032620d1cdf3cd136ff5980df8b3e515cafaf"
LEDGER = UNIT / "supporting/stage3-cumulative-ledger-v1.json"
LEDGER_SHA = "4cbebb9508fc4cb458ec4c93bb2748cfd3bd738510dfd7eb33011bf04cb87174"
OLD_PREFLIGHT = continuation.OUTPUT / "preflight.json"
OLD_PREFLIGHT_SHA = "9a7e44bfcbd36c8515a1af4f0a917c2f44177ef1060a93e84857aa253744b513"
CONTINUATION_SHA = "185c55f23d9aaa2d06dae4f344a44b0c3bb3ebb2dce067caf5f6e46cf96b455d"
SCALE_SHA = "cff77ec4b6ae3f129fa94c1710d7264810dc134cba4091668a2ab738b78238c4"
CASE_ID = "mature:4134:7"
PRIOR_SECONDS = 309.4197279289365
OUTPUT = scale.OUTPUT.parent / "d4-only-v1"


def evidence():
    for path, sha in ((ADMISSION, ADMISSION_SHA), (RULING, RULING_SHA),
                      (LEDGER, LEDGER_SHA), (OLD_PREFLIGHT, OLD_PREFLIGHT_SHA),
                      (Path(continuation.__file__), CONTINUATION_SHA),
                      (Path(scale.__file__), SCALE_SHA)):
        _require(literal_binding(path)["sha256"] == sha, f"D4 bound bytes changed: {path}")
    admission = json.loads(ADMISSION.read_text())
    ruling = json.loads(RULING.read_text())
    ledger = json.loads(LEDGER.read_text())
    receipts = [json.loads(scale.bound(job["receipt"]).read_text()) for job in ledger["jobs"]]
    return admission, ruling, ledger, receipts


def check(admission, ruling, ledger, receipts, case_id):
    _require(admission["status"] == "lead-admitted-D4-only" and
             admission["case_id"] == case_id == CASE_ID and
             admission["budget"]["prior_package_gpu_seconds"] == PRIOR_SECONDS and
             admission["budget"]["remaining_package_seconds"] == 3600-PRIOR_SECONDS and
             admission["budget"]["additional_model_forwards_cap"] == 7 and
             admission["budget"]["additional_vision_forwards_cap"] == 2 and
             admission["budget"]["package_total_model_forwards_cap"] == 50 and
             admission["budget"]["package_total_vision_forwards_cap"] == 18 and
             ledger["cumulative_cost"]["allocated_gpu_seconds"] == PRIOR_SECONDS and
             ledger["cumulative_cost"]["model_forwards"] == 43 and
             ledger["cumulative_cost"]["vision_forwards"] == 16 and
             admission["budget"]["sequence_prior_gpu_hours"] ==
             ledger["cumulative_cost"]["sequence_cumulative_gpu_hours"],
             "D4-only authority or prior charge changed")
    _require(ruling["status"] == "lead-accepted-D1-D2-and-D3-technical-disposition" and
             ruling["accepted_case_ids"] == list(continuation.CASE_IDS[:2]) and
             ruling["unanswered_case_id"] == continuation.CASE_IDS[2] and
             [x["id"] for x in ledger["rows"]] ==
             ["mature:7511:5", *scale.CASE_IDS] and
             [x["status"] for x in ledger["rows"]] ==
             ["lead_accepted"]*3 + ["technical_invalid_unanswered"] +
             ["candidate_complete"]*2 + ["technical_invalid_unanswered",
              "held_unrun_after_D3_technical_failure"] and
             len(receipts) == 8 and
             len(ledger["jobs"]) == 8 and
             [x["status"] for x in receipts] ==
             ["technical_invalid","candidate_complete","candidate_complete",
              "candidate_complete","technical_invalid","candidate_complete",
              "candidate_complete","technical_invalid"],
             "accepted cases or known technical holds changed")
    for index, expected_row in ((4,2),(7,0)):
        failed = receipts[index]
        _require(failed["terminal"] is True and
                 failed["error"] == "ValueError('cached/full native or companion vector mismatch')" and
                 failed["cost"]["model_forwards"] == 3 and
                 failed["cost"]["vision_forwards"] == 2 and
                 len(failed["completed"]) == 1 and
                 ledger["rows"][3 if index == 4 else 6]["failed_row"] == expected_row and
                 ledger["rows"][3 if index == 4 else 6]["gate"] == 2e-4 and
                 ledger["rows"][3 if index == 4 else 6]["full_cached_max_abs_by_row"][expected_row] > 2e-4,
                 "R4/D3 failure or unchanged gate changed")
    _require(all(receipts[i]["cost"]["allocated_gpu_seconds"] ==
                 ledger["jobs"][i]["allocated_gpu_seconds"] for i in range(8)) and
             sum(x["cost"]["allocated_gpu_seconds"] for x in receipts) == PRIOR_SECONDS and
             sum(x["cost"]["model_forwards"] for x in receipts) == 43 and
             sum(x["cost"]["vision_forwards"] for x in receipts) == 16 and
             all(x["terminal"] is True for x in receipts),
             "prior terminal receipts or cumulative charge changed")


def cpu_gate():
    admission, ruling, ledger, receipts = evidence()
    check(admission, ruling, ledger, receipts, CASE_ID)
    checks = []

    def rejects(name, a, r, l, rs, target):
        try: check(a, r, l, rs, target)
        except (ValueError, AssertionError, KeyError, TypeError): checks.append(name)
        else: raise AssertionError(f"D4 caller accepted {name}")

    rejects("other_case_R4",admission,ruling,ledger,receipts,"mature:99184:7")
    changed=copy.deepcopy(receipts);changed[4]["error"]="ValueError('unknown R4')"
    rejects("changed_R4_failure",admission,ruling,ledger,changed,CASE_ID)
    changed=copy.deepcopy(receipts);changed[7]["error"]="ValueError('unknown D3')"
    rejects("changed_D3_failure",admission,ruling,ledger,changed,CASE_ID)
    changed=copy.deepcopy(receipts);changed[7]["cost"].pop("allocated_gpu_seconds")
    rejects("missing_D3_charge",admission,ruling,ledger,changed,CASE_ID)
    changed=copy.deepcopy(admission);changed["budget"]["prior_package_gpu_seconds"]=0
    rejects("missing_prior_charge",changed,ruling,ledger,receipts,CASE_ID)
    changed=copy.deepcopy(ledger);changed["rows"][6]["gate"]=3e-4
    rejects("relaxed_D3_gate",admission,ruling,changed,receipts,CASE_ID)
    return checks


def d4_contract(case_id):
    m, case, receipt, admission = scale.contract(case_id)
    admission = copy.deepcopy(admission)
    admission["budget"]["sequence_cumulative_prior_to_remaining"] = (
        json.loads(ADMISSION.read_text())["budget"]["sequence_prior_gpu_hours"])
    return m, case, receipt, admission


def operation(fn, **overrides):
    source = continuation.borrowed(fn)
    namespace = dict(source.__globals__)
    namespace.update(CASE_IDS=(CASE_ID,), PRIOR_SECONDS=PRIOR_SECONDS,
                     contract=d4_contract, **overrides)
    return FunctionType(source.__code__, namespace, source.__name__,
                        source.__defaults__, source.__closure__)


def preflight(out):
    _require(not out.exists(), "D4 output already exists")
    checks = cpu_gate()
    old = json.loads(OLD_PREFLIGHT.read_text())
    _require(old["status"] == "cpu_qualified_before_gpu" and
             [x["case_id"] for x in old["case_records"]] == list(continuation.CASE_IDS),
             "old qualified four-case source changed")
    for item in old["direct_source_captures"]:
        scale.bound(item["maintained"]); scale.bound(item["capture"])
    record = old["case_records"][-1]
    _, case, _, _ = scale.contract(CASE_ID)
    _require(record["source"] == case["source_bindings"] and
             record["target_index"] == 3 and
             record["full_source_shape"]["width"] == 1445 and
             all(not x["ended_before_source_step"] for x in record["row_crosswalk"]) and
             record["cpu_destination_phase"]["status"] == "passed" and
             {"wrong_target_index","wrong_historical_row_span","wrong_S_sign",
              "companion_content"}.issubset(record["cpu_checks"]),
             "D4 source, companion or CPU verifier changed")
    forecast = old["cost_forecast"]["per_case"][-1]
    authority = json.loads(ADMISSION.read_text())
    _require(forecast["case_id"] == CASE_ID and
             forecast["forecast_seconds_2x"] < authority["budget"]["remaining_package_seconds"] and
             authority["budget"]["sequence_prior_gpu_hours"]+
             forecast["forecast_seconds_2x"]/3600 < 8,
             "D4 forecast exceeds charge-adjusted cap")
    out.mkdir(parents=True)
    sources = [Path(__file__)] + [Path(x["maintained"]["path"]) for x in old["direct_source_captures"]]
    captures = []
    for src in sources:
        rel = src.relative_to(scale.REPO) if src.is_relative_to(scale.REPO) else Path("transformers")/src.name
        saved = preserve_source(src,run_root=out,relative_name=rel)
        captures.append({"maintained":literal_binding(src),"capture":literal_binding(saved)})
    dest = out / "d4-4134-7"
    prefix = ["python","-B","-m","probes.recurrence_dynamics.recurrence_cross_image_phase.d4"]
    commands = {"case_id":CASE_ID,"output":str(dest),
                "gpu":prefix+["run","--case",CASE_ID,"--output",str(dest),"--device","cuda:0"],
                "readback":prefix+["readback","--case",CASE_ID,"--output",str(dest)],
                "reduce":prefix+["reduce","--case",CASE_ID,"--output",str(dest)]}
    packet = {"schema":"recurrence_cross_image_phase.D4_preflight.v1",
              "status":"cpu_qualified_before_gpu","protocol":old["protocol"],
              "manifest":old["manifest"],"admission":old["admission"],
              "r1_producer":old["r1_producer"],"r1_candidate":old["r1_candidate"],
              "case_records":[record],"cost_forecast":{
                  "per_case":[forecast],"remaining_seconds_2x":forecast["forecast_seconds_2x"],
                  "charged_prior_seconds":PRIOR_SECONDS,"package_cap_seconds":3600,
                  "sequence_prior_hours":authority["budget"]["sequence_prior_gpu_hours"]},
              "direct_source_captures":captures,"commands":[commands],
              "D4_admission":literal_binding(ADMISSION),"D3_ruling":literal_binding(RULING),
              "prior_cumulative_ledger":literal_binding(LEDGER),
              "old_preflight":literal_binding(OLD_PREFLIGHT),
              "D4_producer":literal_binding(Path(__file__)),"cpu_rejections":checks}
    _write_new(out/"preflight.json",packet)
    print(json.dumps({"status":packet["status"],"case_id":CASE_ID,
                      "forecast_seconds_2x":forecast["forecast_seconds_2x"],
                      "prior_seconds":PRIOR_SECONDS,"cpu_rejections":checks,
                      "captures":len(captures)}))


def execute(mode, out, case_id, device):
    admission, ruling, ledger, receipts = evidence()
    check(admission, ruling, ledger, receipts, case_id)
    pre = json.loads((out.parent/"preflight.json").read_text())
    _require(pre["D4_admission"] == literal_binding(ADMISSION) and
             pre["D3_ruling"] == literal_binding(RULING) and
             pre["prior_cumulative_ledger"] == literal_binding(LEDGER) and
             pre["D4_producer"] == literal_binding(Path(__file__)) and
             pre["cost_forecast"]["charged_prior_seconds"] == PRIOR_SECONDS and
             out == Path(pre["commands"][0]["output"]),
             "D4 actual caller/output authority changed")
    if mode == "run": operation(scale.run)(out,device,case_id)
    elif mode == "readback": print(json.dumps(operation(scale.cold)(out,case_id)))
    else: operation(scale.reduce,cold=operation(scale.cold))(out,case_id)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","run","readback","reduce"))
    parser.add_argument("--case")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="preflight": preflight(args.output)
    else: execute(args.mode,args.output,args.case,args.device)


if __name__=="__main__": main()
