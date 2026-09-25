"""Four fixed full-prefix continuations after train417044's edge-donut row.

The source, intervention, stop horizon, and readout belong to the unit protocol.
This module only prepares and verifies their original-batch execution.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
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

from probes.training_set_completion.recurrence_book_first_revisit import run as base
from probes.training_set_completion.recurrence_spatial.recurrence_semantics import (
    complete_box_count, parse_rows, EOS, OBJ_START, BOX_END,
)
from src.artifacts.source_provenance import preserve_source


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-25-recurrence-report-quality-feedback"
PROTOCOL = UNIT / "weak-forward-protocol.md"
ACCEPTANCE = UNIT / "lead-cpu-acceptance-v1.json"
SUPPORT = UNIT / "supporting/support-clarification-v2.json"
PROJECTION = UNIT / "supporting/review-projection.json"
PREFLIGHT = UNIT / "supporting/weak-forward-attempt-002-preflight.json"
EARLIER_PREFLIGHT = UNIT / "supporting/weak-forward-preflight-v3.json"
FAILED_RECEIPT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-25-recurrence-report-quality-feedback/weak-forward-v1/controls/receipt.json")
FAILED_FAILURE = UNIT / "supporting/weak-forward-failure-v1.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-25-recurrence-report-quality-feedback/weak-forward-v2")
PANEL = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json")
TARGET, PREFIX, PROMPT, TOL = 3, 49, 1362, 2e-4
ARMS = ("native", "identity_sham", "expanded", "contracted")
BOXES = {"native": (0, 226, 53, 255), "identity_sham": (0, 226, 53, 255),
         "expanded": (0, 216, 68, 259), "contracted": (0, 236, 38, 251)}
IDS = {arm: tuple(151670 + x for x in box) for arm, box in BOXES.items()}
KEYS = ("input_ids", "attention_mask", "position_ids", "cache_position")
REQUEST_IDS = ("coco2017_val_000000016228", "coco2017_train_000000007116",
               "coco2017_train_000000351017", "coco2017_train_000000417044")
require, bind, write_new = base.require, base.bind, base.write_new


def contract():
    require(PROTOCOL.is_file() and ACCEPTANCE.is_file(), "weak-forward authority absent")
    accepted = json.loads(ACCEPTANCE.read_text())
    support = json.loads(SUPPORT.read_text())
    projection = json.loads(PROJECTION.read_text())
    case = next(x for x in projection["cases"] if x["image_id"] == 417044)
    weak = support["weak_candidate"]["failure_forward"]
    require(len(projection["cases"]) == 29 and case["group"] == "refined-03" and
            case["batch_index"] == TARGET and case["j"] == 4 and
            case["rows"][4]["end"] == PREFIX and
            weak["prefix_raw_end"] == PREFIX and weak["coordinate_slots"] == [44, 45, 46, 47] and
            tuple(weak["native_box"]) == BOXES["native"] and
            tuple(weak["clearer_same_owner_box"]) == BOXES["expanded"] and
            tuple(weak["nonimproving_same_owner_box"]) == BOXES["contracted"] and
            tuple(weak["native_coordinate_ids"]) == IDS["native"] and
            tuple(weak["clearer_coordinate_ids"]) == IDS["expanded"] and
            tuple(weak["nonimproving_coordinate_ids"]) == IDS["contracted"] and
            accepted.get("status") is not None, "accepted CPU source/intervention changed")
    for value in case["source_bindings"].values():
        require(bind(value["path"]) == value, "original source/image binding changed")
    return case, weak


def source(q, device):
    case, _ = contract()
    b = case["source_bindings"]
    source_rows = json.loads(Path(b["raw"]["path"]).read_text())["rows"]
    tokens = source_rows[TARGET]["token_ids"]
    boundary = {"group": "refined-03", "batch_index": TARGET, "image_id": 417044,
                "raw_path": b["raw"]["path"], "trace_path": b["trace"]["path"],
                "receipt_path": b["runtime_receipt"]["path"],
                "native_tokens": tokens, "native_token_hash": base.token_hash(tokens)}
    batch, raw, trace, group, planning = base._source(
        boundary, "untied", json.loads(PANEL.read_text()), q, device)
    receipt = json.loads(Path(b["runtime_receipt"]["path"]).read_text())
    require(tuple(batch.request_ids) == REQUEST_IDS and
            [len(x["token_ids"]) for x in raw] == [255, 37, 3084, 3084] and
            list(map(len, batch.prompt_token_ids)) == [1336, 1362, 1362, 1320] and
            int(batch.inputs["pixel_values"].numel()) == 24502272 and
            base.input_identity(batch) == receipt["input_identity"] and
            tokens[39:49] == [151646, 15007, 332, 151647, 151648,
                              *IDS["native"], 151649] and
            raw[1]["token_ids"][36] == EOS and len(raw[1]["token_ids"]) == 37 and
            all(len(raw[j]["token_ids"]) >= PREFIX + 128 for j in (0, 2)),
            "original four-request source/row/EOS shape changed")
    return batch, raw, trace, group, receipt, planning


def written_prefix(raw, arm, *, target=TARGET, slots=(44, 45, 46, 47), box=None):
    require(arm in ARMS and target == TARGET and slots == (44, 45, 46, 47) and
            box in (None, BOXES[arm]), "wrong target/write slots/donor box")
    original = list(raw[TARGET]["token_ids"][:PREFIX])
    require(original[44:48] == list(IDS["native"]), "native row4 source changed")
    result = original.copy()
    if arm != "native":  # sham deliberately traverses the same write path
        result[44:48] = IDS[arm]
    changed = [i for i, (left, right) in enumerate(zip(original, result, strict=True)) if left != right]
    require(changed == ([] if arm in ARMS[:2] else [45, 46, 47]) and
            result[39:44] == original[39:44] and result[48] == 151649 and
            result[44:48] == list(IDS[arm]), "historical write altered source/shape")
    return result


def inputs(model, batch, raw, pad, arm, emitted, *, previous=None,
           target=TARGET, prefix=PREFIX):
    t = len(emitted)
    require(arm in ARMS and target == TARGET and prefix == PREFIX and
            0 <= t < 128 and (t == 0 or emitted[-1] == previous),
            "wrong target/prefix/step/own consumed token")
    tails = base._prefix_tokens(raw, PREFIX + t, pad)
    tails[TARGET] = written_prefix(raw, arm) + list(emitted)
    require(tails[1][:37] == raw[1]["token_ids"] and
            tails[1][37:] == [pad] * (12 + t) and
            all(tails[j] == raw[j]["token_ids"][:PREFIX + t] for j in (0, 2)),
            "original companion source/EOS/pad changed")
    histories = [list(prompt) + tail for prompt, tail in
                 zip(batch.prompt_token_ids, tails, strict=True)]
    full = base.exact_history_inputs(model, batch.inputs, histories,
                                     pad_token_id=pad, logits_to_keep=1)
    width = PROMPT + PREFIX + t
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    require(full["input_ids"].shape == full["attention_mask"].shape == (4, width) and
            full["position_ids"].shape == (3, 4, width) and
            full["input_ids"][:, PROMPT:].tolist() == tails and
            int(full["attention_mask"][:, PROMPT:].sum()) == 4 * (PREFIX + t) and
            all(int(full["attention_mask"][j, :PROMPT].sum()) == len(batch.prompt_token_ids[j])
                for j in range(4)), "full-batch input/three-axis positions changed")
    return full


def stop(emitted):
    require(0 < len(emitted) <= 128, "free-token count outside frozen horizon")
    if emitted[-1] == EOS:
        return "eos"
    if complete_box_count(emitted) >= 8:
        return "eight_complete_rows"
    if len(emitted) == 128:
        return "free_token_cap"
    return None


def caller(model, full, seen):
    seen.update(expected_mask=base.native_4d(full["attention_mask"]),
                expected_input={key: base.tensor_hash(full[key]) for key in KEYS},
                actual_input=None, layers=[], mask_hashes=[],
                rotary_positions_hash=None, history=[], companions=[])
    return model(**full)


def verify_step(model, batch, raw, pad, arm, emitted, full, payload, entry, hidden_size):
    t = len(emitted)
    expected = inputs(model, batch, raw, pad, arm, emitted,
                      previous=emitted[-1] if t else None)
    native = base.native_4d(expected["attention_mask"])
    hashes = {key: base.tensor_hash(expected[key]) for key in KEYS}
    logits = payload["logits"]
    require(all(torch.equal(full[key].detach().cpu(), expected[key].detach().cpu()) for key in KEYS) and
            entry["arm"] == arm and entry["step"] == t and
            entry["own_prefix"] == emitted and
            entry["consumed_previous"] == (emitted[-1] if t else None) and
            entry["actual_input"] == hashes and
            entry["actual_layers"] == list(range(28)) and
            entry["layer_mask_hashes"] == [base.tensor_hash(native)] * 28 and
            entry["native_mask_hash"] == base.tensor_hash(native) and
            entry["rotary_positions_hash"] == base.tensor_hash(expected["position_ids"]) and
            entry["actual_media_shapes"] ==
                {key: list(expected[key].shape) for key in ("pixel_values", "image_grid_thw")} and
            logits.shape == (4, 152670) and torch.isfinite(logits).all().item() and
            entry["chosen"] == int(torch.argmax(logits[TARGET]).item()) and
            payload["companions"].shape == (28, 3, hidden_size) and
            ((t == 0 and "historical_t0" in payload and
              payload["historical_t0"].shape == (28, PREFIX, hidden_size)) or
             (t > 0 and "historical_t0" not in payload)) and
            entry["historical_state_max_abs_vs_t0"] <= TOL,
            "serialized actual input/mask/position/vector/greedy changed")
    return hashes


def require_controls_cold(saved, receipt_binding, reconstructed):
    require(saved["status"] == "cold_pass" and saved["scope"] == "controls" and
            saved["receipt"] == receipt_binding and saved["blocks"] == [reconstructed],
            "prior independent native/sham cold readback changed")


def cpu_checks(batch, raw, pad, text_config, hidden_size):
    model = base.ConfigOnlyRope()
    require(int(text_config.hidden_size) == hidden_size and hidden_size > 0,
            "configured text hidden size changed")
    text_config._attn_implementation = "sdpa"
    checks = []
    for arm in ARMS:
        for t in (0, 1, 9, 127):
            emitted = [151646] + [15007] * (t - 1) if t else []
            full = inputs(model, batch, raw, pad, arm, emitted,
                          previous=emitted[-1] if t else None)
            native = base.native_4d(full["attention_mask"])
            installed = create_causal_mask(text_config, torch.empty((*full["attention_mask"].shape, 1)),
                                          full["attention_mask"], full["cache_position"], None,
                                          position_ids=full["position_ids"][0])
            require(torch.equal(native, installed), "installed native SDPA mask changed")
            seen = {}
            class Fake:
                def __call__(self, **kwargs):
                    seen["actual_input"] = {k: base.tensor_hash(kwargs[k]) for k in KEYS}
                    z = torch.zeros((4, 1, 152670))
                    z[TARGET, 0, 151646] = 1
                    return SimpleNamespace(logits=z)
            result = caller(Fake(), full, seen)
            seen["layers"] = list(range(28))
            seen["mask_hashes"] = [base.tensor_hash(native)] * 28
            seen["rotary_positions_hash"] = base.tensor_hash(full["position_ids"])
            payload = {"logits": result.logits[:, -1],
                       "companions": torch.zeros((28, 3, hidden_size))}
            if t == 0:
                payload["historical_t0"] = torch.zeros((28, PREFIX, hidden_size))
            entry = {"arm": arm, "step": t, "own_prefix": list(emitted),
                     "consumed_previous": emitted[-1] if t else None,
                     "actual_input": seen["actual_input"], "actual_layers": seen["layers"],
                     "layer_mask_hashes": seen["mask_hashes"],
                     "native_mask_hash": base.tensor_hash(native),
                     "rotary_positions_hash": seen["rotary_positions_hash"],
                     "historical_state_max_abs_vs_t0": 0.0,
                     "actual_media_shapes": {key: list(full[key].shape)
                                             for key in ("pixel_values", "image_grid_thw")},
                     "chosen": 151646}
            verify_step(model, batch, raw, pad, arm, emitted, full, payload, entry, hidden_size)
            checks.append(f"caller_{arm}_{t}")
            for name, key, index in (
                ("target", "input_ids", (TARGET, PROMPT + 44)),
                ("own_prefix", "input_ids", (TARGET, PROMPT + PREFIX)) if t else
                    ("row_class", "input_ids", (TARGET, PROMPT + 40)),
                ("companion", "input_ids", (1, PROMPT + 36)),
                ("position", "position_ids", (0, TARGET, PROMPT + 45)),
                ("source_mask", "attention_mask", (1, PROMPT + 36))):
                bad = {**full, key: full[key].clone()}
                bad[key][index] = bad[key][index] + 1
                try:
                    verify_step(model, batch, raw, pad, arm, emitted, bad, payload, entry, hidden_size)
                except ValueError:
                    checks.append("reject_" + name)
                else:
                    raise AssertionError("caller accepted " + name)
            for name, altered in (
                ("false_identity", {**entry, "arm": "native" if arm != "native" else "identity_sham"}),
                ("wrong_mask", {**entry, "layer_mask_hashes": ["bad"] * 28}),
                ("wrong_consumption", {**entry, "consumed_previous": -1}),
                ("wrong_greedy", {**entry, "chosen": 1})):
                try:
                    verify_step(model, batch, raw, pad, arm, emitted, full, payload,
                                json.loads(json.dumps(altered)), hidden_size)
                except ValueError:
                    checks.append("reject_" + name)
                else:
                    raise AssertionError("reader accepted " + name)
        for name, kw in (("target", {"target": 2}), ("slots", {"slots": (44, 45, 46, 48)}),
                         ("box", {"box": BOXES["contracted"] if arm != "contracted" else BOXES["expanded"]})):
            try:
                written_prefix(raw, arm, **kw)
            except ValueError:
                checks.append("reject_write_" + name)
            else:
                raise AssertionError("writer accepted " + name)
    row = [151646, 15007, 332, 151647, 151648, 151670, 151886, 151738, 151929, BOX_END]
    require(complete_box_count(row) == 1 and stop(row) is None and
            complete_box_count(row * 8) == 8 and stop(row * 8) == "eight_complete_rows" and
            stop([EOS]) == "eos" and stop([151646] * 128) == "free_token_cap" and
            complete_box_count([0, BOX_END] + row) == 1 and
            complete_box_count([151646, 15007, 151647, 151648, 152670, BOX_END]) == 0,
            "canonical complete-row/EOS/stray/invalid/cap stop changed")
    checks.append("canonical_variable_row_and_stop_boundaries")
    probe = torch.nn.Identity()
    handles = []
    try:
        def forced(_module, _args):
            raise RuntimeError("forced forward exception")
        handles.append(probe.register_forward_pre_hook(forced))
        try:
            probe(torch.tensor([1]))
        except RuntimeError as exc:
            require(str(exc) == "forced forward exception", "wrong hook exception")
        else:
            raise AssertionError("hook exception not exercised")
    finally:
        for handle in handles:
            handle.remove()
    require(torch.equal(probe(torch.tensor([1])), torch.tensor([1])),
            "hook was not restored after exception")
    checks.append("forced_exception_finally_restoration")
    # The failed native step is qualification-only: no old step receipt or
    # scientific reference is synthesized from these reconstructed fields.
    failed = torch.load(FAILED_RECEIPT.parent / "native-000.pt", map_location="cpu",
                        weights_only=True)
    saved = torch.load(FAILED_RECEIPT.parent / "inputs-native-000.pt", map_location="cpu",
                       weights_only=True)
    expected = inputs(model, batch, raw, pad, "native", [])
    native = base.native_4d(expected["attention_mask"])
    entry = {"arm": "native", "step": 0, "own_prefix": [],
             "consumed_previous": None,
             "actual_input": {key: base.tensor_hash(saved[key]) for key in KEYS},
             "actual_layers": list(range(28)),
             "layer_mask_hashes": [base.tensor_hash(native)] * 28,
             "native_mask_hash": base.tensor_hash(native),
             "rotary_positions_hash": base.tensor_hash(expected["position_ids"]),
             "actual_media_shapes": {key: list(expected[key].shape)
                                     for key in ("pixel_values", "image_grid_thw")},
             "chosen": int(torch.argmax(failed["logits"][TARGET]).item()),
             "historical_state_max_abs_vs_t0": 0.0}
    verify_step(model, batch, raw, pad, "native", [], saved, failed, entry, hidden_size)
    checks.append("saved_failed_step0_shape_qualification_only")
    for name, bad in (
        ("companion_width", {**failed, "companions": failed["companions"][..., :-1]}),
        ("companion_batch", {**failed, "companions": failed["companions"][:, :2]}),
        ("companion_layers", {**failed, "companions": failed["companions"][:-1]}),
        ("historical_width", {**failed, "historical_t0": failed["historical_t0"][..., :-1]}),
        ("historical_length", {**failed, "historical_t0": failed["historical_t0"][:, :-1]}),
        ("historical_layers", {**failed, "historical_t0": failed["historical_t0"][:-1]}),
        ("vector_batch", {**failed, "logits": failed["logits"][:3]})):
        try:
            verify_step(model, batch, raw, pad, "native", [], saved, bad, entry, hidden_size)
        except ValueError:
            checks.append("reject_saved_" + name)
        else:
            raise AssertionError("saved qualification accepted " + name)
    try:
        verify_step(model, batch, raw, pad, "native", [], saved, failed, entry, hidden_size + 1)
    except ValueError:
        checks.append("reject_mismatched_configured_hidden_size")
    else:
        raise AssertionError("verifier accepted mismatched configured hidden size")
    receipt_binding = {"path": "receipt", "sha256": "fixed"}
    reconstructed = {"block": "controls", "arms": [{"arm": "native", "tokens": 1}]}
    cold = {"status": "cold_pass", "scope": "controls", "receipt": receipt_binding,
            "blocks": [reconstructed]}
    require_controls_cold(cold, receipt_binding, reconstructed)
    for name, bad in (("changed_receipt", {**cold, "receipt": {"path": "other"}}),
                      ("changed_reconstruction", {**cold, "blocks": []})):
        try:
            require_controls_cold(bad, receipt_binding, reconstructed)
        except ValueError:
            checks.append("reject_prior_cold_" + name)
        else:
            raise AssertionError("reader accepted " + name)
    return checks


def preflight():
    case, _ = contract()
    require(EARLIER_PREFLIGHT.is_file() and not PREFLIGHT.exists() and
            not (OUT / "controls" / "launch.json").exists(),
            "weak-forward attempt already prepared")
    old_preflight = json.loads(EARLIER_PREFLIGHT.read_text())
    old_source = old_preflight["captures"][0]
    old_failure = json.loads(FAILED_FAILURE.read_text())
    old_receipt = json.loads(FAILED_RECEIPT.read_text())
    require(bind(old_source["capture"]["path"]) == old_source["capture"] and
            old_source["maintained"]["sha256"] == old_source["capture"]["sha256"] and
            old_receipt["status"] == "technical_failure" and
            old_receipt["counts"]["model_forwards"] ==
                old_receipt["counts"]["vision_forwards"] == 1 and
            old_receipt["counts"]["emitted_tokens"] == 0 and
            old_failure["outer_seconds"] == 15.93689356,
            "failed attempt/source capture or charge changed")
    config_path = Path(base.BASE) / "config.json"
    configured = AutoConfig.from_pretrained(base.BASE, local_files_only=True)
    hidden_size = int(configured.text_config.hidden_size)
    require(config_path.is_file() and hidden_size > 0,
            "maintained checkpoint text configuration missing")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU preflight loaded model")
    batch, raw, trace, group, receipt, planning = source(q, torch.device("cpu"))
    pad, special = int(q.tokenizer.pad_token_id), sorted(q.tokenizer.all_special_ids)
    checks = cpu_checks(batch, raw, pad, configured.text_config, hidden_size)
    fixture = base.ConfigOnlyRope()
    widths = [inputs(fixture, batch, raw, pad, "expanded", [151646] * t,
                     previous=151646 if t else None)["input_ids"].shape[1]
              for t in (0, 127)]
    require(widths == [1411, 1538] and int(batch.inputs["pixel_values"].numel()) == 24502272,
            "actual maximum full-prefix source shape changed")
    # Full logits, per-step companion states, input tensors, and compact actual-consumer
    # hashes/receipts. History states are saved once per arm and checked online thereafter.
    per_call = 4 * 152670 * 4 + 28 * 3 * hidden_size * 4 + (4 + 4 + 12) * 1538 * 8 + 150_000
    estimated = 512 * per_call + 4 * 28 * PREFIX * hidden_size * 4 + 256 * 1024**2
    require(estimated < 12 * 1024**3, "artifact retention forecast exceeds 12GiB")
    direct = [Path(__file__), config_path, Path(base.__file__), Path(inspect.getfile(base._source)),
              Path(inspect.getfile(base._prefix_tokens)), Path(inspect.getfile(base._trace_compare)),
              Path(inspect.getfile(base.exact_history_inputs)), Path(inspect.getfile(base.native_4d)),
              Path(inspect.getfile(complete_box_count)), Path(inspect.getfile(parse_rows)),
              Path(inspect.getfile(create_causal_mask)), Path(inspect.getfile(modeling_qwen3_vl)),
              Path(inspect.getfile(base.load_model)), Path(inspect.getfile(preserve_source))]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else Path("installed") / path.name
        saved = preserve_source(path, run_root=OUT / "qualified-source-attempt-002", relative_name=rel)
        captures.append({"maintained": bind(path), "capture": bind(saved)})
    module = "probes.training_set_completion.recurrence_report_quality_feedback.weak_forward"
    commands = {name: ["python", "-B", "-m", module, name] for name in
                ("preflight", "controls", "readback_controls", "treatments", "readback_all", "sheets")}
    packet = {"status": "cpu_qualified_before_model", "protocol": bind(PROTOCOL),
              "superseded_cpu_preflight": bind(EARLIER_PREFLIGHT),
              "old_producer_capture": old_source["capture"],
              "failed_receipt": bind(FAILED_RECEIPT), "failed_failure": bind(FAILED_FAILURE),
              "failed_parent_seconds": old_failure["outer_seconds"],
              "supersession_reason": "lead-authorized hidden-width correction after terminal technical failure",
              "cpu_acceptance": bind(ACCEPTANCE), "cpu_support": bind(SUPPORT),
              "producer": bind(Path(__file__)), "source_bindings": case["source_bindings"],
              "configured_text_hidden_size": hidden_size, "maintained_config": bind(config_path),
              "source_identity": receipt["identity"], "source_input_identity": receipt["input_identity"],
              "request_ids": list(batch.request_ids), "prompt_lengths": list(map(len, batch.prompt_token_ids)),
              "raw_lengths": [len(x["token_ids"]) for x in raw], "pad_id": pad,
              "special_ids": special, "pixel_elements": 24502272, "widths": widths,
              "cpu_checks": checks, "artifact_forecast_bytes": estimated,
              "artifact_envelope_bytes": 12 * 1024**3,
              "rough_parent_seconds": 3443, "prior_sequence_gpu_hours": 0.964693853226186,
              "fresh_max_calls": 512, "failure_inclusive_max_calls": 513,
              "captures": captures, "commands": commands}
    write_new(PREFLIGHT, packet)
    print(json.dumps({"status": packet["status"], "checks": len(checks),
                      "max_width": widths[-1], "artifact_forecast_bytes": estimated}))


def checked():
    contract()
    p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_model" and
            p["protocol"] == bind(PROTOCOL) and p["cpu_acceptance"] == bind(ACCEPTANCE) and
            p["cpu_support"] == bind(SUPPORT) and p["producer"] == bind(Path(__file__)) and
            p["failed_receipt"] == bind(FAILED_RECEIPT) and
            p["failed_failure"] == bind(FAILED_FAILURE) and
            p["old_producer_capture"] ==
                json.loads(EARLIER_PREFLIGHT.read_text())["captures"][0]["capture"] and
            p["maintained_config"] == bind(Path(base.BASE) / "config.json") and
            isinstance(p["configured_text_hidden_size"], int) and
            p["configured_text_hidden_size"] > 0 and
            p["prior_sequence_gpu_hours"] == 0.964693853226186 and
            p["fresh_max_calls"] == 512 and p["failure_inclusive_max_calls"] == 513 and
            p["widths"] == [1411, 1538] and
            p["artifact_forecast_bytes"] < p["artifact_envelope_bytes"],
            "frozen CPU preflight/producer changed")
    for entry in p["captures"]:
        require(bind(entry["maintained"]["path"]) == entry["maintained"] and
                bind(entry["capture"]["path"]) == entry["capture"],
                "maintained import/capture changed")
    return p


def run_block(which):
    p = checked()
    require(which in ("controls", "treatments"), "wrong production block")
    if which == "treatments":
        control = json.loads((OUT / "controls" / "receipt.json").read_text())
        cold = json.loads((OUT / "controls" / "readback.json").read_text())
        require(control["status"] == "raw_complete" and cold["status"] == "cold_pass" and
                cold["receipt"] == bind(OUT / "controls" / "receipt.json"),
                "native/sham cold gate absent")
    root = OUT / which
    require(not root.exists(), "block already launched; no retry")
    root.mkdir(parents=True)
    arms = ARMS[:2] if which == "controls" else ARMS[2:]
    counts = {"model_forwards": 0, "vision_forwards": 0, "emitted_tokens": 0,
              "reused_calls": 0}
    record = {"status": "running", "block": which, "pid": os.getpid(),
              "begun_unix": time.time(), "preflight": bind(PREFLIGHT),
              "producer": bind(Path(__file__)), "counts": counts, "arms": []}
    write_new(root / "launch.json", record)
    handles = []
    active = {}
    started = time.monotonic()
    device = torch.device("cuda:0")
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = base.load_model("untied", device)
        require({k: v for k, v in identity.items() if k != "loader_source"} ==
                {k: v for k, v in p["source_identity"].items() if k != "loader_source"} and
                all(identity["loader_source"][k] == p["source_identity"]["loader_source"][k]
                    for k in ("sha256", "size_bytes")), "effective checkpoint differs")
        model = q.model.eval()
        require(int(model.config.text_config.hidden_size) ==
                p["configured_text_hidden_size"],
                "loaded model text hidden size differs from bound maintained config")
        batch, raw, trace, group, receipt, planning = source(q, device)
        require(receipt["input_identity"] == p["source_input_identity"] and
                int(q.tokenizer.pad_token_id) == p["pad_id"] and
                sorted(q.tokenizer.all_special_ids) == p["special_ids"],
                "GPU source/tokenizer changed")
        pad = p["pad_id"]
        layers = list(model.model.language_model.layers)
        attentions = [x.self_attn for x in layers]
        require(len(attentions) == 28 and all(isinstance(x, modeling_qwen3_vl.Qwen3VLTextAttention)
                                              for x in attentions), "actual attention route changed")
        def top(_module, _args, kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 256, "block model-call cap")
            active["actual_input"] = {key: base.tensor_hash(kwargs[key]) for key in KEYS}
            require(active["actual_input"] == active["expected_input"] and
                    all(kwargs[key] is active["full"][key]
                        for key in ("pixel_values", "image_grid_thw")),
                    "top-level text/media source changed")
            active["actual_media_shapes"] = {
                key: list(kwargs[key].shape) for key in ("pixel_values", "image_grid_thw")}
        def vision(_module, _args):
            counts["vision_forwards"] += 1
            require(counts["vision_forwards"] <= 256, "block vision-call cap")
        def rotary(_module, args, output):
            require(len(args) >= 2 and torch.equal(args[1], active["full"]["position_ids"]),
                    "actual rotary positions changed")
            active["rotary_positions_hash"] = base.tensor_hash(args[1])
        handles.extend((model.register_forward_pre_hook(top, with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(vision),
                        model.model.language_model.rotary_emb.register_forward_hook(rotary)))
        for i, attention in enumerate(attentions):
            def attention_hook(_module, _args, kwargs, *, index=i):
                mask = kwargs.get("attention_mask")
                require(isinstance(mask, torch.Tensor) and mask.ndim == 4 and
                        torch.equal(mask, active["expected_mask"]),
                        f"layer {index} native mask changed")
                active["layers"].append(index)
                active["mask_hashes"].append(base.tensor_hash(mask))
            handles.append(attention.register_forward_pre_hook(attention_hook, with_kwargs=True))
        for i, layer in enumerate(layers):
            def layer_hook(_module, _args, output, *, index=i):
                value = output[0] if isinstance(output, tuple) else output
                require(isinstance(value, torch.Tensor) and value.ndim == 3,
                        "layer states unavailable")
                active["history"].append(value[TARGET, PROMPT:PROMPT+PREFIX].detach().cpu().float())
                active["companions"].append(value[[0, 1, 2], -1].detach().cpu().float())
            handles.append(layer.register_forward_hook(layer_hook))
        record["effective_identity"] = identity
        for arm in arms:
            emitted = []
            arm_record = {"arm": arm, "steps": [], "emitted": [], "stop": None}
            record["arms"].append(arm_record)
            history0 = None
            with torch.inference_mode():
                for t in range(128):
                    previous = emitted[-1] if t else None
                    full = inputs(model, batch, raw, pad, arm, emitted, previous=previous)
                    native_mask = base.native_4d(full["attention_mask"])
                    active.update(full=full)
                    output = caller(model, full, active)
                    torch.cuda.synchronize(device)
                    require(active["layers"] == list(range(28)) and
                            len(active["history"]) == len(active["companions"]) == 28 and
                            active["actual_input"] is not None and
                            active["rotary_positions_hash"] is not None,
                            f"{arm} step{t} actual consumers incomplete")
                    logits = output.logits[:, -1, :].detach().cpu().float()
                    chosen = int(torch.argmax(logits[TARGET]).item())
                    historical = torch.stack(active["history"])
                    companions = torch.stack(active["companions"])
                    if history0 is None:
                        history0 = historical
                    history_error = float((historical - history0).abs().max())
                    require(history_error <= TOL, f"{arm} historical states changed across own suffix")
                    payload = {"logits": logits, "companions": companions}
                    if t == 0:
                        payload["historical_t0"] = historical
                    raw_path = root / f"{arm}-{t:03d}.pt"
                    torch.save(payload, raw_path)
                    input_path = root / f"inputs-{arm}-{t:03d}.pt"
                    torch.save({k: full[k].detach().cpu() for k in KEYS}, input_path)
                    entry = {"arm": arm, "step": t, "own_prefix": list(emitted),
                             "consumed_previous": previous, "raw": bind(raw_path),
                             "inputs": bind(input_path), "actual_input": active["actual_input"],
                             "actual_layers": list(active["layers"]),
                             "layer_mask_hashes": list(active["mask_hashes"]),
                             "native_mask_hash": base.tensor_hash(native_mask),
                             "rotary_positions_hash": active["rotary_positions_hash"],
                             "actual_media_shapes": active["actual_media_shapes"],
                             "historical_state_max_abs_vs_t0": history_error,
                             "chosen": chosen, "internal_seconds": time.monotonic() - started}
                    verify_step(model, batch, raw, pad, arm, emitted, full, payload, entry,
                                p["configured_text_hidden_size"])
                    matches = []
                    for j in (0, 2):
                        match = base._trace_compare(logits=logits[j], trace=trace,
                            batch_index=j, absolute_offset=PREFIX+t,
                            token_id=raw[j]["token_ids"][PREFIX+t], role="active_companion")
                        matches.append(match)
                    require(all(x["passed"] for x in matches), "active companion trace parity failed")
                    entry["active_companion_source_trace"] = matches
                    if arm == "native" and emitted == raw[TARGET]["token_ids"][PREFIX:PREFIX+t]:
                        match = base._trace_compare(logits=logits[TARGET], trace=trace,
                            batch_index=TARGET, absolute_offset=PREFIX+t,
                            token_id=raw[TARGET]["token_ids"][PREFIX+t], role="native_target")
                        require(match["passed"], "native target source trace parity failed")
                        entry["native_target_source_trace"] = match
                    else:
                        entry["native_target_source_trace"] = "UNAVAILABLE_diverged_or_treatment"
                    if arm != "native":
                        reference_root = root if arm == "identity_sham" else OUT / "controls"
                        native_receipt_path = reference_root / "receipt.json"
                        if arm == "identity_sham":
                            native_steps = record["arms"][0]["steps"]
                        else:
                            native_steps = json.loads(native_receipt_path.read_text())["arms"][0]["steps"]
                        if t < len(native_steps):
                            native_payload = torch.load(native_steps[t]["raw"]["path"],
                                                        map_location="cpu", weights_only=True)
                            companion_error = max(
                                float((companions - native_payload["companions"]).abs().max()),
                                max(float((logits[j] - native_payload["logits"][j]).abs().max())
                                    for j in (0, 1, 2)))
                            require(companion_error <= TOL, "matched companion state/vector changed")
                            entry["companion_matched_max_abs"] = companion_error
                            if arm == "identity_sham":
                                all_error = float((logits - native_payload["logits"]).abs().max())
                                history_error_native = float((historical - native_payload["historical_t0"]).abs().max()) if t == 0 else history_error
                                require(max(all_error, history_error_native) <= TOL and
                                        chosen == native_steps[t]["chosen"],
                                        "identity-write sham full-vector/state/greedy mismatch")
                                entry["sham_all_vector_max_abs"] = all_error
                        else:
                            entry["matched_native_vector"] = "UNAVAILABLE_native_horizon_ended"
                    arm_record["steps"].append(entry)
                    emitted.append(chosen)
                    counts["emitted_tokens"] += 1
                    require(counts["model_forwards"] == counts["vision_forwards"] ==
                            counts["emitted_tokens"] <= 256, "block call/token count changed")
                    arm_record["emitted"] = list(emitted)
                    arm_record["complete_rows"] = complete_box_count(emitted)
                    arm_record["stop"] = stop(emitted)
                    write_new(root / f"checkpoint-{arm}-{t:03d}.json", record)
                    if arm_record["stop"] is not None:
                        break
            require(arm_record["stop"] is not None, f"{arm} did not terminate by horizon")
        require([x["arm"] for x in record["arms"]] == list(arms) and
                counts["model_forwards"] == counts["vision_forwards"] == counts["emitted_tokens"] and
                counts["reused_calls"] == 0, "fixed arm order/call counts changed")
        record["status"] = "raw_complete"
    except BaseException as exc:
        record["status"] = "technical_failure"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc(),
                             "active_arm": record["arms"][-1]["arm"] if record["arms"] else None}
    finally:
        for handle in handles:
            handle.remove()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        record["counts"] = dict(counts)
        record["internal_seconds"] = time.monotonic() - started
        record["max_cuda_allocated_bytes"] = (torch.cuda.max_memory_allocated(device)
                                               if torch.cuda.is_available() else None)
        record["artifact_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        write_new(root / "receipt.json", record)
    require(record["status"] == "raw_complete", "terminal production failure; no retry")


def readback(which):
    p = checked()
    require(which in ("controls", "all"), "wrong readback scope")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "cold readback loaded model")
    batch, raw, trace, group, receipt, planning = source(q, torch.device("cpu"))
    model, pad = base.ConfigOnlyRope(), p["pad_id"]
    blocks = ("controls",) if which == "controls" else ("controls", "treatments")
    summary = {"status": "cold_pass", "scope": which, "preflight": bind(PREFLIGHT),
               "blocks": [], "model_loads": 0, "model_forwards": 0,
               "vision_forwards": 0, "cuda_calls": 0}
    native_receipt = json.loads((OUT / "controls" / "receipt.json").read_text())
    native_steps = native_receipt["arms"][0]["steps"]
    for block in blocks:
        root = OUT / block
        r = json.loads((root / "receipt.json").read_text())
        require(r["status"] == "raw_complete" and
                [x["arm"] for x in r["arms"]] == list(ARMS[:2] if block == "controls" else ARMS[2:]),
                "cold block status/arm order changed")
        bsum = {"block": block, "receipt": bind(root / "receipt.json"), "arms": []}
        for item in r["arms"]:
            arm, emitted = item["arm"], []
            for t, entry in enumerate(item["steps"]):
                require(bind(entry["raw"]["path"]) == entry["raw"] and
                        bind(entry["inputs"]["path"]) == entry["inputs"],
                        "cold raw/input binding changed")
                payload = torch.load(entry["raw"]["path"], map_location="cpu", weights_only=True)
                full = torch.load(entry["inputs"]["path"], map_location="cpu", weights_only=True)
                verify_step(model, batch, raw, pad, arm, emitted, full, payload, entry,
                            p["configured_text_hidden_size"])
                if t == 0:
                    require(payload["historical_t0"].shape ==
                            (28, PREFIX, p["configured_text_hidden_size"]),
                            "cold historical t0 evidence changed")
                else:
                    require("historical_t0" not in payload and
                            entry["historical_state_max_abs_vs_t0"] <= TOL,
                            "cold online historical-state gate absent")
                matches = [base._trace_compare(logits=payload["logits"][j], trace=trace,
                    batch_index=j, absolute_offset=PREFIX+t,
                    token_id=raw[j]["token_ids"][PREFIX+t], role="active_companion")
                    for j in (0, 2)]
                require(matches == entry["active_companion_source_trace"] and
                        all(x["passed"] for x in matches),
                        "cold active companion source parity changed")
                if arm == "native" and emitted == raw[TARGET]["token_ids"][PREFIX:PREFIX+t]:
                    match = base._trace_compare(logits=payload["logits"][TARGET], trace=trace,
                        batch_index=TARGET, absolute_offset=PREFIX+t,
                        token_id=raw[TARGET]["token_ids"][PREFIX+t], role="native_target")
                    require(match == entry["native_target_source_trace"] and match["passed"],
                            "cold native target trace changed")
                else:
                    require(entry["native_target_source_trace"] ==
                            "UNAVAILABLE_diverged_or_treatment", "invented native trace")
                if arm != "native" and t < len(native_steps):
                    native_payload = torch.load(native_steps[t]["raw"]["path"],
                                                map_location="cpu", weights_only=True)
                    companion_error = max(
                        float((payload["companions"] - native_payload["companions"]).abs().max()),
                        max(float((payload["logits"][j] - native_payload["logits"][j]).abs().max())
                            for j in (0, 1, 2)))
                    require(companion_error <= TOL and
                            abs(entry["companion_matched_max_abs"] - companion_error) <= 1e-12,
                            "cold matched companion state/vector changed")
                    if arm == "identity_sham":
                        sham_error = float((payload["logits"] - native_payload["logits"]).abs().max())
                        require(sham_error <= TOL and
                                abs(entry["sham_all_vector_max_abs"] - sham_error) <= 1e-12 and
                                entry["chosen"] == native_steps[t]["chosen"],
                                "cold identity sham full-vector/greedy changed")
                elif arm != "native":
                    require(entry["matched_native_vector"] == "UNAVAILABLE_native_horizon_ended",
                            "cold invented native reference beyond horizon")
                emitted.append(entry["chosen"])
            require(emitted == item["emitted"] and stop(emitted) == item["stop"] and
                    complete_box_count(emitted) == item["complete_rows"] and
                    len(emitted) == len(item["steps"]), "cold own-prefix/stop/count changed")
            bsum["arms"].append({"arm": arm, "tokens": len(emitted),
                                 "complete_rows": item["complete_rows"], "stop": item["stop"]})
        require(r["counts"]["model_forwards"] == r["counts"]["vision_forwards"] ==
                r["counts"]["emitted_tokens"] == sum(x["tokens"] for x in bsum["arms"]),
                "cold call/emission ledger changed")
        summary["blocks"].append(bsum)
        path = root / "readback.json"
        if which == "controls" and block == "controls":
            require(not path.exists(), "controls readback already complete; no overwrite")
            summary["receipt"] = bind(root / "receipt.json")
            write_new(path, summary)
        elif block == "controls":
            require_controls_cold(json.loads(path.read_text()),
                                  bind(root / "receipt.json"), bsum)
    if which == "all":
        write_new(OUT / "readback-all.json", summary)
    print(json.dumps({"status": summary["status"], "scope": which,
                      "calls": sum(x["tokens"] for b in summary["blocks"] for x in b["arms"])}))


def sheets():
    p = checked()
    cold = json.loads((OUT / "readback-all.json").read_text())
    require(cold["status"] == "cold_pass", "no complete cold readback")
    q = base.load_qwen_components_from_options(base.QwenLoadOptions(
        base_model=str(base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU sheets loaded model")
    batch, raw, trace, group, receipt, planning = source(q, torch.device("cpu"))
    case, _ = contract()
    image_path = Path(case["image_path"])
    mappings = []
    for index, arm in enumerate(ARMS):
        root = OUT / ("controls" if arm in ARMS[:2] else "treatments")
        r = json.loads((root / "receipt.json").read_text())
        item = next(x for x in r["arms"] if x["arm"] == arm)
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size
        parsed = parse_rows(item["emitted"], q.tokenizer, cell={},
            geometry={"source_width": width, "source_height": height,
                      "canvas_width": width, "canvas_height": height})
        draw = ImageDraw.Draw(image)
        for row in parsed["rows"]:
            if not row.get("source_geometry_valid"):
                continue
            x1, y1, x2, y2 = row["coord_bins_source"]
            box = (round(x1 * (width - 1) / 999), round(y1 * (height - 1) / 999),
                   round(x2 * (width - 1) / 999), round(y2 * (height - 1) / 999))
            draw.rectangle(box, outline="#ff3535", width=3)
            draw.text((box[0], max(0, box[1] - 14)), f"r{row['row_index']}",
                      fill="#ff3535", stroke_width=2, stroke_fill="black")
        name = f"sheet-{index}.png"
        image.save(OUT / name)
        mappings.append({"sheet": name, "arm": arm, "parsed": parsed,
                         "source_image": bind(image_path), "stop": item["stop"],
                         "emitted": item["emitted"]})
    write_new(OUT / "sheet-map.json", mappings)
    print(json.dumps({"sheets": len(mappings)}))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("action", choices=("preflight", "controls", "readback_controls",
                                        "treatments", "readback_all", "sheets"))
    action = cli.parse_args().action
    if action == "preflight": preflight()
    elif action == "controls": run_block("controls")
    elif action == "treatments": run_block("treatments")
    elif action == "readback_controls": readback("controls")
    elif action == "readback_all": readback("all")
    else: sheets()


if __name__ == "__main__":
    main()
