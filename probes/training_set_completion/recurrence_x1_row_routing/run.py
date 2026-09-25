"""Six full-prefix trajectories crossing the first current x1 at train351017."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_collapse_two_record_routing import run as prior
from probes.training_set_completion.recurrence_free_header_routing.run import parse_row
from src.artifacts.source_provenance import preserve_source
from src.data.geometry import iou_xyxy


base = prior.base
require, bind, write_new = prior.require, prior.bind, prior.write_new
ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "research/experiments/2026-09-24-recurrence-x1-row-routing"
PROTOCOL, ADMISSION = UNIT / "unit.md", UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight-v3.json"
PRELIMINARY_PREFLIGHT = UNIT / "supporting/attempt-001-preflight-v2.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-x1-row-routing/attempt-001")
SHAS = {PROTOCOL: "bb5e59d6b306335ed72f6b1ed811e1e346595126099424f120d2c812c13557b6",
        ADMISSION: "e8bef47468960f0d1259943ffd4c2e62553dd651a25b26da88d065a68b5bd15a"}
ARMS = ("AF_native", "FF_native", "AF_sham", "FF_sham", "AF_flip", "FF_flip")
HEADER = (151646, 8987, 151647, 151648)
AF_ROW = (151646, 8987, 151647, 151648, 151670, 151670, 151699, 151756, 151649)
FF_ROW = (151646, 8987, 151647, 151648, 151671, 151683, 152206, 152669, 151649)
TARGET, OFFSET, PROMPT, TOL = 2, 18, 1362, 2e-4
KEYS = ("input_ids", "attention_mask", "position_ids", "cache_position")
AF_TRACE_ROLE, COMP_TRACE_ROLE = "x1_AF_original", "x1_companion"


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen x1 routing authority changed: {path}")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-finite-x1-row-routing" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["source"]["case_id"] == "mature:351017:2" and
            a["source"]["group"] == "refined-03" and
            a["source"]["target_index"] == TARGET and
            a["source"]["history_raw_end"] == OFFSET and
            a["source"]["A_tokens"] == list(prior.ROW0) and
            a["source"]["F_tokens"] == list(prior.ROW1) and
            a["source"]["native_row2_tokens"] == list(AF_ROW) and
            [x["name"] for x in a["arms"]] == list(ARMS) and
            [x["history"] for x in a["arms"]] == ["AF", "FF"] * 3 and
            [x["override_token"] for x in a["arms"]] ==
                [None, None, 151670, 151671, 151671, 151670] and
            [x["max_tokens"] for x in a["arms"]] == [9, 9, 9, 9, 16, 16] and
            a["current_override"]["row_step"] == 4 and
            a["current_override"]["target_raw_index"] == 22 and
            a["current_override"]["target_physical_index"] == 1384 and
            a["current_override"]["native_header"] == list(HEADER) and
            a["counts"]["max_model_forwards"] == a["counts"]["max_vision_forwards"] ==
                a["counts"]["max_logical_emitted_tokens"] == 68 and
            a["counts"]["reused_calls"] == 0 and
            a["counts"]["explicit_x1_writes"] == 4 and
            a["prior_sequence_gpu_hours"] == 0.7265508404969789 and
            a["owned_paths"][-1] == str(OUT), "finite x1 routing cells/counts changed")
    for key in ("protocol", "predecessor_acceptance", "predecessor_receipt",
                "predecessor_readback", "predecessor_verification", "latest_acceptance",
                "source_panel"):
        require(bind(a[key]["path"]) == a[key], f"bound predecessor {key} changed")
    for key in ("raw", "trace", "runtime_receipt", "image"):
        require(bind(a["source_bindings"][key]["path"]) == a["source_bindings"][key],
                f"original source {key} changed")
    cross = a["loader_crosswalk"]
    require(bind(cross["maintained"]["path"]) == cross["maintained"] and
            cross["maintained"]["sha256"] == cross["historical"]["sha256"] and
            bind(cross["prior_acceptance"]["path"]) == cross["prior_acceptance"],
            "maintained loader crosswalk changed")
    for origin, row in (("AF", AF_ROW), ("FF", FF_ROW)):
        ref = a["references"][origin]
        require(ref["tokens"] == list(row) and len(ref["steps"]) == 9,
                "same-base nine-step reference changed")
        for step, entry in enumerate(ref["steps"]):
            require(entry["step"] == step and entry["chosen"] == row[step] and
                    bind(entry["raw"]["path"]) == entry["raw"] and
                    bind(entry["inputs"]["path"]) == entry["inputs"],
                    f"same-base {origin} reference step {step} changed")
    old_a = prior.contract()
    require(a["source_bindings"]["input_identity_sha256"] ==
            old_a["source_bindings"]["input_identity_sha256"] and
            a["expected_source_shape"] ==
            {"prompt_width": 1362, "prefix_width": 1380, "maximum_width": 1395,
             "pixel_elements": 24502272, "original_raw_lengths": [255, 37, 3084, 3084],
             "max_companion_source_offset": 33} and
            a["reference_tolerance"] == TOL and
            a["primary"]["A"] == [0, 13, 536, 999] and
            a["primary"]["F"] == [0, 0, 33, 86] and
            a["primary"]["same_region_iou_min"] == .5 and
            a["primary"]["other_region_iou_max"] == .1 and
            a["primary"]["numerical_guard"] == 1e-6 and
            a["secondary"]["AF_flip_free_suffix"] == list(FF_ROW[5:]) and
            a["secondary"]["FF_flip_free_suffix"] == list(AF_ROW[5:]),
            "source geometry or frozen endpoint changed")
    return a, old_a


def step_inputs(model, batch, raw, pad, config, emitted, *, previous=None,
                target=TARGET, prefix=OFFSET, history=None):
    t = len(emitted)
    require(config["name"] in ARMS and target == TARGET and prefix == OFFSET and
            history in (None, config["history"]) and
            0 <= t < config["max_tokens"] and
            (t == 0 or emitted[-1] == previous), "wrong arm/target/history/step/own token")
    raw_history = list(raw[TARGET]["token_ids"][:OFFSET])
    require(raw_history == list(prior.ROW0 + prior.ROW1), "source historical rows changed")
    if config["history"] == "FF":
        raw_history[5:8] = prior.ROW1[5:8]
    tails = base._prefix_tokens(raw, OFFSET + t, pad)
    tails[TARGET] = raw_history + list(emitted)
    histories = [list(prompt) + tail for prompt, tail in
                 zip(batch.prompt_token_ids, tails, strict=True)]
    full = base.exact_history_inputs(model, batch.inputs, histories,
                                     pad_token_id=pad, logits_to_keep=1)
    width = PROMPT + OFFSET + t
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    require(full["input_ids"].shape == full["attention_mask"].shape == (4, width) and
            full["position_ids"].shape == (3, 4, width) and
            full["input_ids"][:, PROMPT:].tolist() == tails and
            tails[TARGET][:OFFSET] == raw_history and
            tails[TARGET][OFFSET:] == list(emitted) and
            all(tails[j] == raw[j]["token_ids"][:OFFSET+t] for j in (0, 1, 3)) and
            (t != 0 or tails[TARGET] == raw_history) and
            all(int(full["attention_mask"][j, :PROMPT].sum()) ==
                len(batch.prompt_token_ids[j]) for j in range(4)),
            "full-prefix history/companion/current prefix/position changed")
    return full


def select_token(config, t, emitted, logits, raw_argmax=None, selected=None):
    require(config["name"] in ARMS and 0 <= t < config["max_tokens"] and
            logits.shape == (4, 152670) and torch.isfinite(logits).all().item(),
            "policy arm/step/vector changed")
    greedy = int(torch.argmax(logits[TARGET]).item())
    if t <= 4:
        require(list(emitted) == list((AF_ROW if config["history"] == "AF" else FF_ROW)[:t]) and
                greedy == (AF_ROW if config["history"] == "AF" else FF_ROW)[t],
                "pre-fork header/native x1 qualification failed")
    wanted = config["override_token"] if t == 4 and config["override_token"] is not None else greedy
    require((raw_argmax is None or raw_argmax == greedy) and
            (selected is None or selected == wanted) and
            (t != 4 or list(emitted) == list(HEADER)),
            "raw greedy or exact x1 policy selection changed")
    return greedy, wanted, ("identity_write" if t == 4 and config["override_token"] == greedy
                             else "flip_write" if t == 4 and config["override_token"] is not None
                             else "greedy")


def caller(model, full, seen):
    native = base.native_4d(full["attention_mask"])
    seen.update(expected_mask=native, actual_input=None, layers=[],
                layer_mask_hashes=[], history=[], companions=[], rotary_positions=None)
    return model(**full)


def verify_step(model, batch, raw, pad, config, emitted, full, seen, logits,
                raw_argmax, selected, *, previous=None):
    expected = step_inputs(model, batch, raw, pad, config, emitted, previous=previous)
    hashes = {key: base.tensor_hash(expected[key]) for key in KEYS}
    native = base.native_4d(expected["attention_mask"])
    greedy, chosen, policy = select_token(config, len(emitted), emitted, logits,
                                          raw_argmax=raw_argmax, selected=selected)
    require(all(torch.equal(full[key], expected[key]) for key in KEYS) and
            seen["actual_input"] == hashes and
            seen["layers"] == list(range(28)) and
            seen["layer_mask_hashes"] == [base.tensor_hash(native)] * 28 and
            torch.equal(seen["expected_mask"], native) and
            seen["rotary_positions"] == base.tensor_hash(expected["position_ids"]) and
            (not emitted or int(full["input_ids"][TARGET, PROMPT+OFFSET+len(emitted)-1]) ==
             emitted[-1]), "actual source/28-layer mask/rotary/own-token consumer changed")
    return hashes, policy


def receipt_guard(records, a):
    require(isinstance(records, list) and [x["arm"] for x in records] == list(ARMS),
            "serialized arm order/container changed")
    for record, config in zip(records, a["arms"], strict=True):
        require(record["history"] == config["history"] and
                record["override_token"] == config["override_token"] and
                isinstance(record["steps"], list) and
                [x["step"] for x in record["steps"]] ==
                list(range(len(record["steps"]))) and
                1 <= len(record["steps"]) <= config["max_tokens"],
                "serialized arm history/policy/step order changed")


def cpu_checks(a, batch, raw, pad, special):
    fixture = base.ConfigOnlyRope()
    checks = []
    require(all(parse_row(list(row), special)["stop"] == "complete"
                for row in (prior.ROW0, prior.ROW1, AF_ROW, FF_ROW)) and
            parse_row([151645], special)["stop"] == "eos" and
            parse_row([151649], special)["stop"] == "early_row_terminator" and
            parse_row([151646, 8987, 151647, 151648, 152670], special)["stop"] ==
                "malformed_coordinate" and
            parse_row([151646, 8987, 151647, 151648, 152669], special)["stop"] is None and
            parse_row([151646] + [8987] * 15, special)["stop"] == "cap" and
            parse_row([151646, 8987, 151647, 151648, 151670, 151671,
                       151672, 151673, 151670], special)["stop"] == "malformed_terminator",
            "first-row parser/exclusive coordinate boundary changed")
    checks.append("parser_complete_eos_early_malformed_exclusive_cap")
    cfg = AutoConfig.from_pretrained(base.BASE, local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    for config in a["arms"]:
        for t in (0, 4, 5, 8, 15):
            if t >= config["max_tokens"]:
                continue
            emitted = list((AF_ROW if config["history"] == "AF" else FF_ROW)[:t])
            if t > 4 and config["override_token"] is not None:
                emitted[4] = config["override_token"]
            if t == 15:
                emitted = list(HEADER) + [config["override_token"] or 151670] + [151670] * 10
            full = step_inputs(fixture, batch, raw, pad, config, emitted,
                               previous=emitted[-1] if t else None)
            native = base.native_4d(full["attention_mask"])
            installed = create_causal_mask(
                cfg, torch.empty((*full["attention_mask"].shape, 1)),
                full["attention_mask"], full["cache_position"], None,
                position_ids=full["position_ids"][0])
            require(torch.equal(native, installed), "installed native SDPA mask changed")
            seen = {}
            class Fake:
                def __call__(self, **kwargs):
                    seen["actual_input"] = {key: base.tensor_hash(kwargs[key]) for key in KEYS}
                    z = torch.zeros((4, 1, 152670))
                    z[TARGET, 0, (AF_ROW if config["history"] == "AF" else FF_ROW)[t]
                      if t <= 4 else 151670] = 1
                    return SimpleNamespace(logits=z)
            logits = caller(Fake(), full, seen).logits[:, -1]
            seen["layers"] = list(range(28))
            seen["layer_mask_hashes"] = [base.tensor_hash(native)] * 28
            seen["rotary_positions"] = base.tensor_hash(full["position_ids"])
            greedy, selected, _ = select_token(config, t, emitted, logits)
            verify_step(fixture, batch, raw, pad, config, emitted, full, seen,
                        logits, greedy, selected, previous=emitted[-1] if t else None)
            checks.append(f"actual_caller_{config['name']}_step{t}")
            for label, key, idx in (
                    ("historical_extra_write", "input_ids", (TARGET, PROMPT+6)),
                    ("current_header", "input_ids", (TARGET, PROMPT+OFFSET)) if t else
                        ("historical_class", "input_ids", (TARGET, PROMPT+1)),
                    ("companion", "input_ids", (0, PROMPT+OFFSET-1)),
                    ("position", "position_ids", (0, TARGET, PROMPT+5)),
                    ("source_mask", "attention_mask", (0, 0))):
                bad = {**full, key: full[key].clone()}
                bad[key][idx] += 1
                try:
                    verify_step(fixture, batch, raw, pad, config, emitted, bad, seen,
                                logits, greedy, selected, previous=emitted[-1] if t else None)
                except ValueError:
                    checks.append("reject_" + label)
                else:
                    raise AssertionError("actual caller accepted " + label)
            if t == 5:
                for label, wrong in (("dropped_supplied_x1", emitted[:-1]),
                                     ("replaced_supplied_x1", emitted[:-1] + [1 - (emitted[-1]-151670) + 151670])):
                    try:
                        verify_step(fixture, batch, raw, pad, config, wrong, full, seen,
                                    logits, greedy, selected, previous=wrong[-1])
                    except ValueError:
                        checks.append("reject_" + label)
                    else:
                        raise AssertionError("actual caller accepted " + label)
            for label, kw in (("wrong_target", {"target": 3}),
                              ("wrong_history", {"history": "FF" if config["history"] == "AF" else "AF"})):
                try:
                    step_inputs(fixture, batch, raw, pad, config, emitted,
                                previous=emitted[-1] if t else None, **kw)
                except ValueError:
                    checks.append("reject_" + label)
                else:
                    raise AssertionError("source caller accepted " + label)
            if t:
                try:
                    step_inputs(fixture, batch, raw, pad, config, emitted, previous=-1)
                except ValueError:
                    checks.append("reject_wrong_step_previous")
                else:
                    raise AssertionError("source caller accepted wrong previous token")
            if t == 4:
                for label, raw_bad, selected_bad in (
                        ("wrong_raw_argmax", greedy+1, selected),
                        ("wrong_policy_token", greedy, greedy if selected != greedy else greedy+1)):
                    try:
                        select_token(config, t, emitted, logits,
                                     raw_argmax=raw_bad, selected=selected_bad)
                    except ValueError:
                        checks.append("reject_" + label)
                    else:
                        raise AssertionError("selection caller accepted " + label)
    good = [{"arm": x["name"], "history": x["history"],
             "override_token": x["override_token"],
             "steps": [{"step": j} for j in range(x["max_tokens"])]} for x in a["arms"]]
    receipt_guard(json.loads(json.dumps(good)), a)
    for label, bad in (("swapped", good[:4] + [good[5], good[4]]),
                       ("missing", good[:-1]),
                       ("wrong_write", good[:4] + [{**good[4], "override_token": None}] + good[5:]),
                       ("step_order", good[:4] + [{**good[4], "steps": good[4]["steps"][1:]}] + good[5:])):
        try:
            receipt_guard(json.loads(json.dumps(bad)), a)
        except ValueError:
            checks.append("serialized_reject_" + label)
        else:
            raise AssertionError("serialized consumer accepted " + label)
    return checks


def preflight():
    a, old_a = contract()
    require(not PREFLIGHT.exists() and not (OUT / "launch.json").exists(),
            "attempt already prepared/launched")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU preflight loaded language model")
    batch, raw, trace, source_receipt, _ = prior.source(q, old_a, torch.device("cpu"))
    pad, special = int(q.tokenizer.pad_token_id), frozenset(q.tokenizer.all_special_ids)
    checks = cpu_checks(a, batch, raw, pad, special)
    fixture = base.ConfigOnlyRope()
    widths = []
    for t in range(16):
        emitted = list(HEADER) + [151670] * (t-4) if t >= 4 else list(HEADER[:t])
        full = step_inputs(fixture, batch, raw, pad, a["arms"][4], emitted,
                           previous=emitted[-1] if t else None)
        widths.append(int(full["input_ids"].shape[1]))
    require(widths == list(range(1380, 1396)) and
            int(batch.inputs["pixel_values"].numel()) == 24502272 and
            source_receipt["identity"]["attention"] == "sdpa" and
            a["source_bindings"]["input_identity_sha256"] ==
                old_a["source_bindings"]["input_identity_sha256"],
            "CPU source geometry/model/input binding changed")
    for origin, row in (("AF", AF_ROW), ("FF", FF_ROW)):
        for t, ref in enumerate(a["references"][origin]["steps"]):
            expected = step_inputs(fixture, batch, raw, pad,
                                   a["arms"][0 if origin == "AF" else 1],
                                   list(row[:t]), previous=row[t-1] if t else None)
            saved = json.loads(Path(ref["inputs"]["path"]).read_text())
            require(all(saved[key] == expected[key].cpu().tolist() for key in KEYS),
                    f"saved {origin} same-base input step {t} differs")
            payload = torch.load(ref["raw"]["path"], map_location="cpu", weights_only=True)
            require(payload["logits"].shape == (4, 152670) and
                    torch.isfinite(payload["logits"]).all().item(),
                    f"saved {origin} full vector step {t} invalid")
    checks.append("all18_bound_reference_inputs_and_full_vectors")
    measured = json.loads(Path(a["predecessor_receipt"]["path"]).read_text())
    require(measured["counts"]["model_forwards"] == 45 and
            measured["counts"]["vision_forwards"] == 45 and
            measured["artifact_bytes"] > 0 and
            a["forecast"]["basis_calls"] == 45 and
            a["forecast"]["max_width"] == max(widths),
            "measured full-prefix basis changed")
    forecast_seconds = (2 * a["forecast"]["basis_outer_seconds"] *
                        (68 / 45) * (max(widths) / a["forecast"]["basis_max_width"]))
    artifact_forecast = (2 * measured["artifact_bytes"] * (68 / 45) +
                         256 * 1024**2)
    require(forecast_seconds > 0 and artifact_forecast <
            a["forecast"]["artifact_planning_bytes"] == 3 * 1024**3,
            "full-prefix call/artifact reforecast exceeds envelope")
    from probes.training_set_completion.recurrence_free_header_routing import run as free
    prior_pre = json.loads(prior.PREFLIGHT.read_text())
    free_pre = json.loads(free.PREFLIGHT.read_text())
    direct = [Path(__file__), Path(prior.__file__), Path(free.__file__),
              Path(base.__file__), Path(inspect.getfile(create_causal_mask)),
              Path(inspect.getfile(modeling_qwen3_vl)),
              Path(inspect.getfile(iou_xyxy)),
              Path(inspect.getfile(preserve_source))]
    for packet in (prior_pre, free_pre):
        direct += [Path(x["maintained"]["path"]) for x in packet["direct_source_captures"]]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("installed") / path.name
        saved = preserve_source(path, run_root=OUT / "qualified-preflight-v3",
                                relative_name=rel)
        captures.append({"maintained": bind(path), "capture": bind(saved)})
    command = ["python", "-B", "-m",
               "probes.training_set_completion.recurrence_x1_row_routing.run"]
    packet = {"status": "cpu_qualified_before_model", "protocol": bind(PROTOCOL),
              "admission": bind(ADMISSION), "producer": bind(Path(__file__)),
              "superseded_cpu_preflight": bind(PRELIMINARY_PREFLIGHT),
              "supersession_reason": "cold overlay skips invalid geometry without suppressing its saved outcome; no model call occurred",
              "source_identity": source_receipt["identity"],
              "source_input_identity": source_receipt["input_identity"],
              "request_ids": list(batch.request_ids), "pad_id": pad,
              "special_ids": sorted(special), "pixel_elements": 24502272,
              "widths": widths, "reference_steps": 18, "checks": checks,
              "direct_source_captures": captures,
              "forecast_outer_seconds": forecast_seconds,
              "forecast_artifact_bytes": artifact_forecast,
              "artifact_envelope_bytes": a["forecast"]["artifact_planning_bytes"],
              "prior_sequence_gpu_hours": a["prior_sequence_gpu_hours"],
              "commands": {action: command + [action] for action in
                           ("preflight", "run", "gpu", "readback")}}
    write_new(PREFLIGHT, packet)
    print(json.dumps({"status": packet["status"], "checks": len(checks),
                      "captures": len(captures), "forecast_outer_seconds": forecast_seconds,
                      "forecast_artifact_bytes": artifact_forecast}))


def checked():
    a, old_a = contract()
    p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_model" and
            p["protocol"] == bind(PROTOCOL) and
            p["admission"] == bind(ADMISSION) and
            p["superseded_cpu_preflight"] == bind(PRELIMINARY_PREFLIGHT) and
            p["producer"] == bind(Path(__file__)) and
            p["reference_steps"] == 18 and p["widths"] == list(range(1380, 1396)),
            "frozen CPU preflight/producer changed")
    for entry in p["direct_source_captures"]:
        require(bind(entry["maintained"]["path"]) == entry["maintained"] and
                bind(entry["capture"]["path"]) == entry["capture"],
                "direct maintained source/capture changed")
    return a, old_a, p


def run_parent():
    checked()
    require(not (OUT / "launch.json").exists() and not (OUT / "outer.json").exists(),
            "attempt already launched; no retry")
    OUT.mkdir(parents=True, exist_ok=True)
    command = ["python", "-B", "-m",
               "probes.training_set_completion.recurrence_x1_row_routing.run", "gpu"]
    began, unix = time.monotonic(), time.time()
    with (OUT / "stdout.log").open("x") as stdout, (OUT / "stderr.log").open("x") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        code = child.wait()
    outer = {"command": command, "started_unix": unix,
             "outer_seconds": time.monotonic()-began,
             "child_pid": child.pid, "returncode": code, "terminal": True,
             "stdout": bind(OUT / "stdout.log"), "stderr": bind(OUT / "stderr.log")}
    write_new(OUT / "outer.json", outer)
    print(json.dumps(outer))
    require(code == 0, "terminal child failure; no automatic retry")


def gpu_child():
    a, old_a, p = checked()
    require(not (OUT / "launch.json").exists() and not (OUT / "receipt.json").exists(),
            "attempt already launched; no retry")
    OUT.mkdir(parents=True, exist_ok=True)
    began, device = time.monotonic(), torch.device("cuda:0")
    handles = []
    counts = {"model_forwards": 0, "vision_forwards": 0,
              "logical_emitted_tokens": 0, "greedy_selections": 0,
              "identity_x1_writes": 0, "counterfactual_x1_writes": 0,
              "reused_calls": 0}
    receipt = {"status": "running", "pid": os.getpid(), "begun_unix": time.time(),
               "protocol": bind(PROTOCOL), "admission": bind(ADMISSION),
               "preflight": bind(PREFLIGHT), "producer": bind(Path(__file__)),
               "counts": counts, "arms": []}
    write_new(OUT / "launch.json", receipt)
    active = {}
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = base.load_model("untied", device)
        expected_identity = p["source_identity"]
        require({k: v for k, v in identity.items() if k != "loader_source"} ==
                {k: v for k, v in expected_identity.items() if k != "loader_source"} and
                all(identity["loader_source"][key] == expected_identity["loader_source"][key]
                    for key in ("sha256", "size_bytes")) and
                identity["loader_source"]["path"] ==
                a["loader_crosswalk"]["maintained"]["path"],
                "effective checkpoint/maintained loader changed")
        model = q.model.eval()
        batch, raw, trace, source_receipt, _ = prior.source(q, old_a, device)
        require(source_receipt["input_identity"] == p["source_input_identity"] and
                int(q.tokenizer.pad_token_id) == p["pad_id"] and
                sorted(q.tokenizer.all_special_ids) == p["special_ids"],
                "GPU original full batch/tokenizer changed")
        pad, special = p["pad_id"], frozenset(p["special_ids"])
        layers = list(model.model.language_model.layers)
        attentions = [x.self_attn for x in layers]
        require(len(layers) == len(attentions) == 28 and
                all(isinstance(x, modeling_qwen3_vl.Qwen3VLTextAttention)
                    for x in attentions), "actual 28-layer attention route changed")
        def top(_module, _args, kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 68, "model-forward cap exceeded")
            active["actual_input"] = {key: base.tensor_hash(kwargs[key]) for key in KEYS}
            require(active["actual_input"] == active["expected_input_hashes"],
                    "top-level model input changed")
        def vision(_module, _args):
            counts["vision_forwards"] += 1
            require(counts["vision_forwards"] <= 68, "vision-forward cap exceeded")
        def rotary(_module, args, output):
            require(len(args) >= 2 and torch.equal(args[1], active["full"]["position_ids"]),
                    "actual three-axis rotary positions changed")
            active["rotary_positions"] = base.tensor_hash(args[1])
        handles.extend((model.register_forward_pre_hook(top, with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(vision),
                        model.model.language_model.rotary_emb.register_forward_hook(rotary)))
        for i, attention in enumerate(attentions):
            def at_attention(_module, _args, kwargs, *, index=i):
                mask = kwargs.get("attention_mask")
                require(isinstance(mask, torch.Tensor) and mask.ndim == 4 and
                        torch.equal(mask, active["expected_mask"]),
                        f"layer {index} native mask changed")
                active["layers"].append(index)
                active["layer_mask_hashes"].append(base.tensor_hash(mask))
            handles.append(attention.register_forward_pre_hook(at_attention, with_kwargs=True))
        for i, layer in enumerate(layers):
            def at_layer(_module, _args, output, *, index=i):
                value = output[0] if isinstance(output, tuple) else output
                require(isinstance(value, torch.Tensor) and value.ndim == 3,
                        "historical/companion states unavailable")
                active["history"].append(value[TARGET, PROMPT:PROMPT+OFFSET].detach().cpu().float())
                active["companions"].append(value[[0, 1, 3], -1].detach().cpu().float())
            handles.append(layer.register_forward_hook(at_layer))
        receipt["effective_identity"] = identity
        native_payloads = {"AF": [], "FF": []}
        with torch.inference_mode():
            for config in a["arms"]:
                name, origin = config["name"], config["history"]
                emitted = []
                record = {"arm": name, "history": origin,
                          "override_token": config["override_token"],
                          "steps": [], "emitted": [], "stop": None}
                receipt["arms"].append(record)
                for t in range(config["max_tokens"]):
                    previous = emitted[-1] if t else None
                    full = step_inputs(model, batch, raw, pad, config, emitted,
                                       previous=previous)
                    expected_mask = base.native_4d(full["attention_mask"])
                    active.update(full=full, expected_mask=expected_mask,
                                  expected_input_hashes={key: base.tensor_hash(full[key])
                                                         for key in KEYS})
                    result = caller(model, full, active)
                    torch.cuda.synchronize(device)
                    require(active["layers"] == list(range(28)) and
                            len(active["history"]) == len(active["companions"]) == 28 and
                            active["actual_input"] is not None and
                            active["rotary_positions"] is not None,
                            f"{name} step{t} actual consumer incomplete")
                    logits = result.logits[:, -1, :].detach().cpu().float()
                    greedy, selected, policy = select_token(config, t, emitted, logits)
                    input_hashes, verified_policy = verify_step(
                        model, batch, raw, pad, config, emitted, full, active, logits,
                        greedy, selected, previous=previous)
                    require(policy == verified_policy, "selection policy/consumer differs")
                    payload = {"arm": name, "step": t, "logits": logits,
                               "actual_mask": expected_mask.detach().cpu(),
                               "historical_by_layer": torch.stack(active["history"]),
                               "companions_by_layer": torch.stack(active["companions"])}
                    path = OUT / f"{name}-step{t}.pt"
                    torch.save(payload, path)
                    input_path = OUT / f"inputs-{name}-step{t}.json"
                    write_new(input_path, {key: full[key].detach().cpu().tolist() for key in KEYS})
                    entry = {"step": t, "raw": bind(path), "inputs": bind(input_path),
                             "actual_input": input_hashes,
                             "actual_layers": list(active["layers"]),
                             "layer_mask_hashes": list(active["layer_mask_hashes"]),
                             "native_mask_hash": base.tensor_hash(expected_mask),
                             "rotary_positions_hash": active["rotary_positions"],
                             "raw_argmax": greedy, "selected": selected, "policy": policy,
                             "consumed_previous": previous,
                             "internal_seconds": time.monotonic()-began}
                    record["steps"].append(entry)
                    if name == "AF_native":
                        parity = [base._trace_compare(
                            logits=logits[j], trace=trace, batch_index=j,
                            absolute_offset=OFFSET+t, token_id=raw[j]["token_ids"][OFFSET+t],
                            role=AF_TRACE_ROLE) for j in range(4)]
                        require(all(x["passed"] for x in parity), "AF source trace parity failed")
                        entry["source_trace_parity"] = parity
                    else:
                        parity = [base._trace_compare(
                            logits=logits[j], trace=trace, batch_index=j,
                            absolute_offset=OFFSET+t, token_id=raw[j]["token_ids"][OFFSET+t],
                            role=COMP_TRACE_ROLE) for j in (0, 1, 3)]
                        require(all(x["passed"] for x in parity),
                                "companion source trace parity failed")
                        entry["companion_trace_parity"] = parity
                    if name in ARMS[:4] or t <= 4:
                        reference = a["references"][origin]["steps"][t]
                        accepted = torch.load(reference["raw"]["path"], map_location="cpu",
                                              weights_only=True)["logits"]
                        error = float((logits - accepted).abs().max())
                        require(error <= TOL and
                                json.loads(input_path.read_text()) ==
                                json.loads(Path(reference["inputs"]["path"]).read_text()),
                                f"{name} step{t} same-base accepted reference changed: {error}")
                        entry["accepted_same_base_max_abs"] = error
                    native = native_payloads[origin]
                    if name in ARMS[:2]:
                        native.append(payload)
                    else:
                        baseline = native[0]
                        historical_error = float((payload["historical_by_layer"] -
                                                  baseline["historical_by_layer"]).abs().max())
                        require(historical_error <= TOL,
                                f"{name} changed prior target states: {historical_error}")
                        entry["historical_state_max_abs"] = historical_error
                        if t < len(native):
                            matched = native[t]
                            companion_error = max(
                                float((payload["companions_by_layer"] -
                                       matched["companions_by_layer"]).abs().max()),
                                max(float((logits[j] - matched["logits"][j]).abs().max())
                                    for j in (0, 1, 3)))
                            require(companion_error <= TOL,
                                    f"{name} matched companion state/vector changed")
                            entry["companion_matched_max_abs"] = companion_error
                            if name in ARMS[2:4]:
                                sham_error = max(
                                    float((payload["historical_by_layer"] -
                                           matched["historical_by_layer"]).abs().max()),
                                    float((logits - matched["logits"]).abs().max()))
                                require(sham_error <= TOL,
                                        f"{name} identity-write sham state/vector changed")
                                entry["sham_same_base_max_abs"] = sham_error
                        else:
                            entry["matched_native_full_vector_state"] = "UNAVAILABLE_not_executed"
                    emitted.append(selected)
                    counts["logical_emitted_tokens"] += 1
                    if policy == "greedy":
                        counts["greedy_selections"] += 1
                    elif policy == "identity_write":
                        counts["identity_x1_writes"] += 1
                    else:
                        counts["counterfactual_x1_writes"] += 1
                    require(counts["logical_emitted_tokens"] <= 68 and
                            counts["model_forwards"] == counts["vision_forwards"] ==
                            counts["logical_emitted_tokens"], "finite call/token counts changed")
                    parsed = parse_row(emitted, special)
                    record["emitted"], record["stop"] = list(emitted), parsed["stop"]
                    receipt["counts"] = dict(counts)
                    write_new(OUT / f"checkpoint-{name}-{t}.json", receipt)
                    if parsed["stop"] is not None:
                        break
                require(record["stop"] is not None, f"{name} did not terminate by cap")
                if name in ARMS[:4]:
                    require(record["stop"] == "complete" and
                            record["emitted"] == list(AF_ROW if origin == "AF" else FF_ROW),
                            f"{name} native/identity-sham row qualification failed")
        require([x["arm"] for x in receipt["arms"]] == list(ARMS) and
                counts["identity_x1_writes"] == counts["counterfactual_x1_writes"] == 2 and
                counts["model_forwards"] == counts["vision_forwards"] ==
                counts["logical_emitted_tokens"] <= 68 and counts["reused_calls"] == 0,
                "six-arm finite counts/write order changed")
        receipt_guard(receipt["arms"], a)
        receipt["status"] = "candidate_raw_complete"
    except BaseException as exc:
        receipt["status"] = "technical_failure"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc(),
                              "active_arm": active.get("arm")}
    finally:
        for handle in handles:
            handle.remove()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        receipt["counts"] = dict(counts)
        receipt["internal_seconds"] = time.monotonic()-began
        receipt["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0)
        receipt["gpu_peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0)
        receipt["artifact_bytes_before_receipt"] = sum(
            x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        receipt["terminal_pid"] = os.getpid()
        write_new(OUT / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "counts": counts,
                          "failure": receipt.get("failure", {}).get("message")}))
    require(receipt["status"] == "candidate_raw_complete",
            "technical failure; no automatic retry")


def readback():
    a, old_a, p = checked()
    require(not (OUT / "readback.json").exists() and
            not (UNIT / "candidate-results.md").exists(),
            "candidate readback already exists")
    receipt = json.loads((OUT / "receipt.json").read_text())
    outer = json.loads((OUT / "outer.json").read_text())
    require(receipt["status"] == "candidate_raw_complete" and
            receipt["protocol"] == bind(PROTOCOL) and
            receipt["admission"] == bind(ADMISSION) and
            receipt["preflight"] == bind(PREFLIGHT) and
            receipt["producer"] == bind(Path(__file__)) and
            receipt["counts"]["model_forwards"] ==
            receipt["counts"]["vision_forwards"] ==
            receipt["counts"]["logical_emitted_tokens"] <= 68 and
            receipt["counts"]["reused_calls"] == 0 and
            receipt["counts"]["identity_x1_writes"] ==
            receipt["counts"]["counterfactual_x1_writes"] == 2 and
            outer["terminal"] and outer["returncode"] == 0 and
            outer["child_pid"] == receipt["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists() and
            all(bind(outer[key]["path"]) == outer[key] for key in ("stdout", "stderr")),
            "terminal six-arm job/count/source binding changed")
    receipt_guard(receipt["arms"], a)
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "cold reader loaded language model")
    batch, raw, trace, source_receipt, _ = prior.source(q, old_a, torch.device("cpu"))
    require(source_receipt["input_identity"] == p["source_input_identity"] and
            int(q.tokenizer.pad_token_id) == p["pad_id"] and
            sorted(q.tokenizer.all_special_ids) == p["special_ids"],
            "cold original full batch/tokenizer changed")
    pad, special = p["pad_id"], frozenset(p["special_ids"])
    fixture = base.ConfigOnlyRope()
    count = {"model_forwards": 0, "vision_forwards": 0,
             "logical_emitted_tokens": 0, "greedy_selections": 0,
             "identity_x1_writes": 0, "counterfactual_x1_writes": 0,
             "reused_calls": 0}
    result = {"status": "candidate_cold_readback_passed", "protocol": bind(PROTOCOL),
              "admission": bind(ADMISSION), "preflight": bind(PREFLIGHT),
              "producer": bind(Path(__file__)), "receipt": bind(OUT / "receipt.json"),
              "outer": bind(OUT / "outer.json"), "arms": [],
              "model_loads": 0, "model_forwards": 0, "vision_forwards": 0,
              "cuda_calls": 0, "gpu_seconds_added_by_readback": 0}
    native_payloads = {"AF": [], "FF": []}
    for record, config in zip(receipt["arms"], a["arms"], strict=True):
        name, origin = record["arm"], config["history"]
        emitted, summaries = [], []
        for t, entry in enumerate(record["steps"]):
            require(entry["step"] == t and
                    entry["actual_layers"] == list(range(28)) and
                    entry["consumed_previous"] == (emitted[-1] if t else None) and
                    bind(entry["raw"]["path"]) == entry["raw"] and
                    bind(entry["inputs"]["path"]) == entry["inputs"],
                    f"cold {name} step{t} raw/input/order binding changed")
            full = {key: torch.tensor(value, dtype=torch.long) for key, value in
                    json.loads(Path(entry["inputs"]["path"]).read_text()).items()}
            payload = torch.load(entry["raw"]["path"], map_location="cpu", weights_only=True)
            logits = payload["logits"]
            require(payload["arm"] == name and payload["step"] == t and
                    logits.shape == (4, 152670) and torch.isfinite(logits).all().item() and
                    payload["historical_by_layer"].shape[0] ==
                    payload["companions_by_layer"].shape[0] == 28,
                    f"cold {name} step{t} full-vector/states changed")
            expected = step_inputs(fixture, batch, raw, pad, config, emitted,
                                   previous=emitted[-1] if t else None)
            native_mask = base.native_4d(expected["attention_mask"])
            require(all(torch.equal(full[key], expected[key]) for key in KEYS) and
                    torch.equal(payload["actual_mask"], native_mask) and
                    entry["native_mask_hash"] == base.tensor_hash(native_mask) and
                    entry["layer_mask_hashes"] ==
                    [base.tensor_hash(native_mask)] * 28 and
                    entry["rotary_positions_hash"] ==
                    base.tensor_hash(expected["position_ids"]),
                    f"cold {name} step{t} source/native mask/rotary changed")
            seen = {"actual_input": entry["actual_input"],
                    "layers": entry["actual_layers"],
                    "layer_mask_hashes": entry["layer_mask_hashes"],
                    "expected_mask": native_mask,
                    "rotary_positions": entry["rotary_positions_hash"]}
            hashes, policy = verify_step(
                fixture, batch, raw, pad, config, emitted, full, seen, logits,
                entry["raw_argmax"], entry["selected"],
                previous=emitted[-1] if t else None)
            require(hashes == entry["actual_input"] and policy == entry["policy"],
                    f"cold {name} step{t} selection/consumer changed")
            if name == "AF_native":
                parity = [base._trace_compare(
                    logits=logits[j], trace=trace, batch_index=j,
                    absolute_offset=OFFSET+t, token_id=raw[j]["token_ids"][OFFSET+t],
                    role=AF_TRACE_ROLE) for j in range(4)]
                require(all(x["passed"] for x in parity) and
                        entry["source_trace_parity"] == parity,
                        f"cold {name} step{t} original trace changed")
            else:
                parity = [base._trace_compare(
                    logits=logits[j], trace=trace, batch_index=j,
                    absolute_offset=OFFSET+t, token_id=raw[j]["token_ids"][OFFSET+t],
                    role=COMP_TRACE_ROLE) for j in (0, 1, 3)]
                require(all(x["passed"] for x in parity) and
                        entry["companion_trace_parity"] == parity,
                        f"cold {name} step{t} companion source trace changed")
            if name in ARMS[:4] or t <= 4:
                ref = a["references"][origin]["steps"][t]
                accepted = torch.load(ref["raw"]["path"], map_location="cpu",
                                      weights_only=True)["logits"]
                error = float((logits - accepted).abs().max())
                require(error <= TOL and
                        json.loads(Path(entry["inputs"]["path"]).read_text()) ==
                        json.loads(Path(ref["inputs"]["path"]).read_text()) and
                        abs(entry["accepted_same_base_max_abs"] - error) <= 1e-12,
                        f"cold {name} step{t} same-base reference changed")
            if name in ARMS[:2]:
                native_payloads[origin].append(payload)
            else:
                baseline = native_payloads[origin][0]
                error = float((payload["historical_by_layer"] -
                               baseline["historical_by_layer"]).abs().max())
                require(error <= TOL and
                        abs(entry["historical_state_max_abs"] - error) <= 1e-12,
                        f"cold {name} step{t} historical states changed")
                if t < len(native_payloads[origin]):
                    matched = native_payloads[origin][t]
                    companion_error = max(
                        float((payload["companions_by_layer"] -
                               matched["companions_by_layer"]).abs().max()),
                        max(float((logits[j] - matched["logits"][j]).abs().max())
                            for j in (0, 1, 3)))
                    require(companion_error <= TOL and
                            abs(entry["companion_matched_max_abs"] -
                                companion_error) <= 1e-12,
                            f"cold {name} step{t} matched companion changed")
                    if name in ARMS[2:4]:
                        sham_error = max(
                            float((payload["historical_by_layer"] -
                                   matched["historical_by_layer"]).abs().max()),
                            float((logits-matched["logits"]).abs().max()))
                        require(sham_error <= TOL and
                                abs(entry["sham_same_base_max_abs"]-sham_error) <= 1e-12,
                                f"cold {name} step{t} identity sham changed")
                else:
                    require(entry["matched_native_full_vector_state"] ==
                            "UNAVAILABLE_not_executed",
                            "cold unexecuted same-step native comparator invented")
            z = logits[TARGET].double()
            top = torch.topk(z, 2)
            summaries.append({
                "step": t, "raw_argmax": entry["raw_argmax"],
                "selected": entry["selected"], "policy": policy,
                "top2_ids": top.indices.tolist(), "top2_logits": top.values.tolist(),
                "log_normalizer": float(torch.logsumexp(z, -1)),
                "selected_logprob": float(torch.log_softmax(z, -1)[entry["selected"]]),
                "raw": entry["raw"], "inputs": entry["inputs"]})
            emitted.append(entry["selected"])
            count["model_forwards"] += 1
            count["vision_forwards"] += 1
            count["logical_emitted_tokens"] += 1
            count[{"greedy": "greedy_selections",
                   "identity_write": "identity_x1_writes",
                   "flip_write": "counterfactual_x1_writes"}[policy]] += 1
            stop = parse_row(emitted, special)["stop"]
            require(stop is None if t < len(record["steps"])-1 else stop == record["stop"],
                    f"cold {name} parser/stop changed")
        require(emitted == record["emitted"] and
                (name not in ARMS[:4] or
                 (record["stop"] == "complete" and
                  emitted == list(AF_ROW if origin == "AF" else FF_ROW))),
                f"cold {name} complete trajectory changed")
        parsed = parse_row(emitted, special)
        box = [x-151670 for x in parsed["box_ids"]] if len(parsed["box_ids"]) == 4 else None
        geometry = ("valid" if box[0] < box[2] and box[1] < box[3]
                    else "invalid") if box else "no_complete_box"
        result["arms"].append({
            "arm": name, "history": origin, "emitted": emitted,
            "raw_argmax_tokens": [x["raw_argmax"] for x in record["steps"]],
            "policy_selected_tokens": emitted, "stop": record["stop"],
            "description_ids": parsed["description_ids"],
            "description": q.tokenizer.decode(
                parsed["description_ids"], skip_special_tokens=False,
                clean_up_tokenization_spaces=False),
            "box": box, "geometry": geometry, "steps": summaries})
    require(count == receipt["counts"] and len(result["arms"]) == 6 and
            0 < count["model_forwards"] <= 68 and
            count["identity_x1_writes"] == count["counterfactual_x1_writes"] == 2,
            "cold six-arm counts/write policy changed")
    result["counts"] = count
    A, F = a["primary"]["A"], a["primary"]["F"]
    def region(row):
        if row["stop"] != "complete":
            return row["stop"]
        if row["description_ids"] != [8987]:
            return "other_class"
        if row["geometry"] != "valid":
            return "invalid_geometry"
        ia, iff = iou_xyxy(row["box"], A), iou_xyxy(row["box"], F)
        row["iou_A"], row["iou_F"] = ia, iff
        if min(abs(ia-.5), abs(iff-.1), abs(iff-.5), abs(ia-.1)) <= 1e-6:
            return "numerical_HOLD"
        if ia >= .5 and iff <= .1:
            return "broad_A"
        if iff >= .5 and ia <= .1:
            return "fragment_F"
        return "neither_region"
    by_name = {row["arm"]: row for row in result["arms"]}
    for row in result["arms"]:
        row["region"] = region(row)
    components = {"AF_flip_broad_A": by_name["AF_flip"]["region"] == "broad_A",
                  "FF_flip_fragment_F": by_name["FF_flip"]["region"] == "fragment_F"}
    result["primary_components"] = components
    result["shared_primary_pass"] = all(components.values())
    secondary = {"AF_flip_free_suffix": by_name["AF_flip"]["emitted"][5:] ==
                 a["secondary"]["AF_flip_free_suffix"],
                 "FF_flip_free_suffix": by_name["FF_flip"]["emitted"][5:] ==
                 a["secondary"]["FF_flip_free_suffix"]}
    result["secondary_components"] = secondary
    result["shared_secondary_pass"] = all(secondary.values())
    image = Image.open(a["source_bindings"]["image"]["path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, box, color in (
            ("A historical", A, "#00ee77"), ("F historical", F, "#ff5533"),
            ("AF native", by_name["AF_native"]["box"], "#eeb000"),
            ("FF native", by_name["FF_native"]["box"], "#0099ff"),
            ("AF flip", by_name["AF_flip"]["box"], "#ee00dd"),
            ("FF flip", by_name["FF_flip"]["box"], "#ffffff")):
        if box is None or box[0] >= box[2] or box[1] >= box[3]:
            continue
        xy = [round(v * (image.width if j % 2 == 0 else image.height) / 1000)
              for j, v in enumerate(box)]
        draw.rectangle(xy, outline=color, width=4)
        draw.text((xy[0], max(0, xy[1]-14)), label, fill=color)
    image.save(OUT / "overlay.png")
    result["overlay"] = bind(OUT / "overlay.png")
    result["prior_sequence_gpu_hours"] = a["prior_sequence_gpu_hours"]
    result["outer_seconds"] = outer["outer_seconds"]
    result["charged_sequence_gpu_hours"] = (
        a["prior_sequence_gpu_hours"] + outer["outer_seconds"]/3600)
    result["limit"] = ("Numerical regions are not physical owner identities; "
                       "F/F2 remains HOLD. No natural onset or mediation fraction.")
    write_new(OUT / "readback.json", result)
    lines = ["# First-x1 row-routing candidate", "",
             "Status: **candidate cold readback passed; lead acceptance pending.**", "",
             f"Protocol SHA `{SHAS[PROTOCOL]}`; admission SHA `{SHAS[ADMISSION]}`; "
             f"producer SHA `{bind(Path(__file__))['sha256']}`.",
             f"Receipt SHA `{result['receipt']['sha256']}`; cold readback SHA "
             f"`{bind(OUT/'readback.json')['sha256']}`. Original refined-03 four requests, "
             "target2 train351017. AF/FF history differs only at older raw5/6/7; all "
             "current headers were freely emitted.", "", "## Frozen outcomes", "",
             f"Shared reciprocal primary **{result['shared_primary_pass']}**; "
             f"AF broad-A **{components['AF_flip_broad_A']}**, FF fragment-F "
             f"**{components['FF_flip_fragment_F']}**. Exact free-suffix secondary "
             f"AF **{secondary['AF_flip_free_suffix']}**, FF "
             f"**{secondary['FF_flip_free_suffix']}**, conjunction "
             f"**{result['shared_secondary_pass']}**. Supplied x1 has no prediction credit.", "",
             "| Arm | Complete selected row | Stop | Box | IoU A | IoU F | Region |",
             "|---|---|---|---|---:|---:|---|"]
    for row in result["arms"]:
        lines.append(
            f"| {row['arm']} | {','.join(map(str,row['emitted']))} | {row['stop']} | "
            f"{row['box']} | {row.get('iou_A','—')} | {row.get('iou_F','—')} | "
            f"{row['region']} |")
    lines += ["", "Raw greedy and selected tokens, every full-vocabulary vector, "
              "input and stepwise log-normalizer are in the cold readback. The original "
              "image overlay is bound there. F/F2 physical owner remains HOLD.", "",
              "## Qualification and cost", "",
              "Both nine-step natives and both independently executed identity writes "
              "passed same-base saved input/vector and source gates before either flip. "
              "Flip steps0–4 matched same-base native references before selection. "
              "Actual next-forward consumption, native masks at all28 layers, rotary "
              "positions, historical and companion states, source trace, parser, "
              "terminal jobs and all raw vectors passed separate CPU readback. "
              "Beyond nine steps, source companion traces were checked without "
              "inventing a native same-step full vector.", "",
              f"Fresh {count['model_forwards']} model / {count['vision_forwards']} vision / "
              f"{count['logical_emitted_tokens']} logical emitted, of which "
              f"{count['greedy_selections']} greedy, {count['identity_x1_writes']} "
              f"identity writes and {count['counterfactual_x1_writes']} flips; zero reuse. "
              f"Parent outer {outer['outer_seconds']:.9f}s, internal "
              f"{receipt['internal_seconds']:.9f}s. Prior sequence "
              f"{a['prior_sequence_gpu_hours']:.12f}GPUh, charged cumulative "
              f"{result['charged_sequence_gpu_hours']:.12f}GPUh.",
              f"Peak RSS {receipt['rss_peak_kib']}KiB; GPU allocated/reserved "
              f"{receipt['gpu_peak_allocated_bytes']}/{receipt['gpu_peak_reserved_bytes']}B; "
              f"raw bytes before receipt {receipt['artifact_bytes_before_receipt']}; "
              f"terminal PID {outer['child_pid']}, exit {outer['returncode']}.", "",
              f"Raw attempt: `{OUT}`. This is a conditional first-token intervention, "
              "not a physical new-owner, natural onset or internal-pathway claim. "
              "No self-acceptance or successor.", ""]
    (UNIT / "candidate-results.md").write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "counts": count,
                      "rows": {x["arm"]: x["emitted"] for x in result["arms"]},
                      "regions": {x["arm"]: x["region"] for x in result["arms"]},
                      "shared_primary": result["shared_primary_pass"],
                      "outer_seconds": outer["outer_seconds"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run", "gpu", "readback"))
    {"preflight": preflight, "run": run_parent,
     "gpu": gpu_child, "readback": readback}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
