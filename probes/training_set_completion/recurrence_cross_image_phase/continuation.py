"""Bound D-only continuation after the accepted R4 technical hold."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from types import FunctionType

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_cross_image_phase import scale
from probes.training_set_completion.recurrence_first_arrivals.prepare import _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from src.artifacts.source_provenance import preserve_source

UNIT = scale.UNIT
ADMISSION = UNIT / "lead-D-continuation-v1.json"
ADMISSION_SHA = "b81588f28bda98c82a949a133be0faa1e4facdf99680b13b8f9cbbc2889c1b59"
RULING = UNIT / "lead-r2-r3-acceptance-and-r4-ruling.json"
RULING_SHA = "e64d460697e4bf08302eba33ccc0a42820760ee1d679f16b1b110902f0a93d73"
LEDGER = UNIT / "supporting/scale-terminal-ledger-v1.json"
LEDGER_SHA = "cad84062b5d6291ecbba4bbc078db4f4c80ca8dd557e11869b60714f4f974f1c"
OLD_PREFLIGHT = scale.OUTPUT / "preflight.json"
OLD_PREFLIGHT_SHA = "a4fa0966317618e341a57996710c36b4f55e21b9caf8bbf429bce59ebf9b02ca"
SCALE_SHA = "cff77ec4b6ae3f129fa94c1710d7264810dc134cba4091668a2ab738b78238c4"
CASE_IDS = ("mature:1584:2", "mature:2299:2", "mature:2685:12", "mature:4134:7")
PRIOR_SECONDS = 186.57946596294641
OUTPUT = scale.OUTPUT.parent / "d-continuation-v1"


def evidence():
    for path, sha in ((ADMISSION, ADMISSION_SHA), (RULING, RULING_SHA),
                      (LEDGER, LEDGER_SHA), (OLD_PREFLIGHT, OLD_PREFLIGHT_SHA),
                      (Path(scale.__file__), SCALE_SHA)):
        _require(literal_binding(path)["sha256"] == sha, f"continuation source changed: {path}")
    admission = json.loads(ADMISSION.read_text())
    ruling = json.loads(RULING.read_text())
    ledger = json.loads(LEDGER.read_text())
    receipts = [json.loads(scale.bound(row["receipt"]).read_text()) for row in ledger["rows"][:4]]
    return admission, ruling, ledger, receipts


def check(admission, ruling, ledger, receipts, case_id):
    _require(admission["status"] == "lead-admitted-D-only-continuation" and
             tuple(admission["case_ids"]) == CASE_IDS and
             case_id in CASE_IDS and
             admission["budget"]["charged_package_gpu_seconds"] == PRIOR_SECONDS and
             admission["budget"]["additional_model_forwards_cap"] == 28 and
             admission["budget"]["additional_vision_forwards_cap"] == 8 and
             admission["budget"]["remaining_package_seconds"] == 3600-PRIOR_SECONDS and
             admission["budget"]["sequence_prior_gpu_hours"] ==
             ledger["cost"]["sequence_cumulative_gpu_hours"] and
             ledger["cost"]["package_gpu_seconds"] == PRIOR_SECONDS and
             ledger["cost"]["total_package_calls_including_prior_failure"] ==
             {"model":26,"vision":10},
             "D-only authority/order/prior cost changed")
    _require(ruling["status"] == "lead-accepted-R2-R3-and-R4-technical-disposition" and
             ruling["accepted_case_ids"] == list(scale.CASE_IDS[0:2]) and
             ruling["unanswered_case_id"] == scale.CASE_IDS[2] and
             [x["id"] for x in ledger["rows"]] ==
             ["mature:7511:5", *scale.CASE_IDS] and
             [x["status"] for x in ledger["rows"][:4]] ==
             ["lead_accepted","candidate_complete","candidate_complete","technical_invalid"] and
             all(x["status"] == "held_unrun_after_R4_technical_failure"
                 for x in ledger["rows"][4:]) and len(receipts) == 4,
             "prior case order/status or R4 HOLD changed")
    for i in range(3):
        _require(receipts[i]["status"] == "candidate_complete" and
                 receipts[i]["cost"]["model_forwards"] == 7 and
                 receipts[i]["cost"]["vision_forwards"] == 2 and
                 (abs(ledger["rows"][0]["allocated_gpu_seconds"] -
                      receipts[0]["cost"]["allocated_gpu_seconds"] -
                      14.241613768041134) < 1e-6 if i == 0 else
                  receipts[i]["cost"]["allocated_gpu_seconds"] ==
                  ledger["rows"][i]["allocated_gpu_seconds"]),
                 "accepted prior receipt changed")
    r4 = receipts[3]
    _require(r4["status"] == "technical_invalid" and r4["terminal"] is True and
             r4["error"] == "ValueError('cached/full native or companion vector mismatch')" and
             r4["cost"]["allocated_gpu_seconds"] == ledger["rows"][3]["allocated_gpu_seconds"] and
             r4["cost"]["model_forwards"] == 3 and r4["cost"]["vision_forwards"] == 2 and
             len(r4["completed"]) == 1 and
             ledger["rows"][3]["full_cached_max_abs_by_row"][2] > 2e-4 and
             ledger["rows"][3]["full_cached_max_abs_by_row"][1] < 2e-4 and
             ledger["queue_stop"]["no_treatments_or_later_cases_launched"] is True and
             sum(x["cost"]["allocated_gpu_seconds"] for x in receipts[1:]) +
             ledger["rows"][0]["allocated_gpu_seconds"] == PRIOR_SECONDS,
             "known R4 failure, receipt or charged cost changed")


def cpu_gate():
    admission, ruling, ledger, receipts = evidence()
    for case_id in CASE_IDS:
        check(admission, ruling, ledger, receipts, case_id)
    checks = []

    def rejects(name, a, r, l, rs, case_id):
        try: check(a, r, l, rs, case_id)
        except (ValueError, AssertionError, KeyError, TypeError): checks.append(name)
        else: raise AssertionError(f"continuation caller accepted {name}")

    changed = copy.deepcopy(receipts); changed[3]["error"] = "ValueError('unknown failure')"
    rejects("unknown_R4_failure", admission, ruling, ledger, changed, CASE_IDS[0])
    changed = copy.deepcopy(receipts); changed[3]["cost"].pop("allocated_gpu_seconds")
    rejects("missing_R4_charge", admission, ruling, ledger, changed, CASE_IDS[0])
    changed = copy.deepcopy(admission); changed["case_ids"] = list(reversed(CASE_IDS))
    rejects("changed_D_order", changed, ruling, ledger, receipts, CASE_IDS[0])
    rejects("R4_replay", admission, ruling, ledger, receipts, scale.CASE_IDS[2])
    changed = copy.deepcopy(ledger); changed["cost"]["package_gpu_seconds"] = 0
    rejects("missing_prior_package_charge", admission, ruling, changed, receipts, CASE_IDS[0])
    return checks


def continuation_contract(case_id):
    m, case, receipt, admission = scale.contract(case_id)
    admission = copy.deepcopy(admission)
    admission["budget"]["sequence_cumulative_prior_to_remaining"] = (
        json.loads(ADMISSION.read_text())["budget"]["sequence_prior_gpu_hours"])
    return m, case, receipt, admission


def borrowed(fn, **overrides):
    namespace = dict(fn.__globals__)
    namespace.update(CASE_IDS=CASE_IDS, PRIOR_SECONDS=PRIOR_SECONDS,
                     contract=continuation_contract, **overrides)
    return FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)


def preflight(out):
    _require(not out.exists(), "D continuation output already exists")
    checks = cpu_gate()
    old = json.loads(OLD_PREFLIGHT.read_text())
    _require(old["status"] == "cpu_qualified_before_gpu" and
             [x["case_id"] for x in old["case_records"][3:]] == list(CASE_IDS),
             "old qualified D source cases changed")
    for item in old["direct_source_captures"]:
        scale.bound(item["maintained"]); scale.bound(item["capture"])
    records = old["case_records"][3:]
    for case_id, record in zip(CASE_IDS, records, strict=True):
        _, case, _, _ = scale.contract(case_id)
        _require(record["target_index"] == case["batch_index"] and
                 record["source"] == case["source_bindings"] and
                 record["cpu_destination_phase"]["status"] == "passed" and
                 "wrong_target_index" in record["cpu_checks"] and
                 "wrong_historical_row_span" in record["cpu_checks"],
                 f"old CPU batch qualification changed: {case_id}")
    values = old["cost_forecast"]["per_case"][3:]
    total = sum(x["forecast_seconds_2x"] for x in values)
    admission = json.loads(ADMISSION.read_text())
    _require(total < admission["budget"]["remaining_package_seconds"] and
             admission["budget"]["sequence_prior_gpu_hours"]+total/3600 < 8,
             "D shape-aware forecast exceeds charge-adjusted cap")
    out.mkdir(parents=True)
    sources = [Path(__file__)] + [Path(x["maintained"]["path"]) for x in old["direct_source_captures"]]
    captures = []
    for src in sources:
        rel = src.relative_to(scale.REPO) if src.is_relative_to(scale.REPO) else Path("transformers")/src.name
        saved = preserve_source(src, run_root=out, relative_name=rel)
        captures.append({"maintained":literal_binding(src),"capture":literal_binding(saved)})
    prefix = ["python","-B","-m","probes.training_set_completion.recurrence_cross_image_phase.continuation"]
    commands = []
    for i, case_id in enumerate(CASE_IDS, 1):
        dest = out / f"d{i}-{case_id.split(':')[1]}-{case_id.split(':')[2]}"
        commands.append({"case_id":case_id,"output":str(dest),
                         "gpu":prefix+["run","--case",case_id,"--output",str(dest),"--device","cuda:0"],
                         "readback":prefix+["readback","--case",case_id,"--output",str(dest)],
                         "reduce":prefix+["reduce","--case",case_id,"--output",str(dest)]})
    packet = {"schema":"recurrence_cross_image_phase.D_continuation_preflight.v1",
              "status":"cpu_qualified_before_gpu","protocol":old["protocol"],
              "manifest":old["manifest"],"admission":old["admission"],
              "r1_producer":old["r1_producer"],"r1_candidate":old["r1_candidate"],
              "case_records":records,"cost_forecast":{
                  "per_case":values,"remaining_seconds_2x":total,
                  "charged_prior_seconds":PRIOR_SECONDS,"package_cap_seconds":3600,
                  "sequence_prior_hours":admission["budget"]["sequence_prior_gpu_hours"]},
              "direct_source_captures":captures,"commands":commands,
              "D_continuation":literal_binding(ADMISSION),"R4_ruling":literal_binding(RULING),
              "prior_terminal_ledger":literal_binding(LEDGER),
              "old_preflight":literal_binding(OLD_PREFLIGHT),
              "continuation_producer":literal_binding(Path(__file__)),"cpu_rejections":checks}
    _write_new(out/"preflight.json",packet)
    print(json.dumps({"status":packet["status"],"cases":len(records),
                      "forecast_seconds_2x":total,"prior_seconds":PRIOR_SECONDS,
                      "cpu_rejections":checks,"captures":len(captures)}))


def execute(mode, out, case_id, device):
    admission, ruling, ledger, receipts = evidence()
    check(admission, ruling, ledger, receipts, case_id)
    pre = json.loads((out.parent/"preflight.json").read_text())
    _require(pre["D_continuation"] == literal_binding(ADMISSION) and
             pre["R4_ruling"] == literal_binding(RULING) and
             pre["prior_terminal_ledger"] == literal_binding(LEDGER) and
             pre["continuation_producer"] == literal_binding(Path(__file__)) and
             pre["cost_forecast"]["charged_prior_seconds"] == PRIOR_SECONDS,
             "D continuation launch authority changed")
    if mode == "run": borrowed(scale.run)(out, device, case_id)
    elif mode == "readback": print(json.dumps(borrowed(scale.cold)(out, case_id)))
    else:
        cold = borrowed(scale.cold)
        borrowed(scale.reduce, cold=cold)(out, case_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight","run","readback","reduce"))
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "preflight": preflight(args.output)
    else: execute(args.mode, args.output, args.case, args.device)


if __name__ == "__main__": main()
