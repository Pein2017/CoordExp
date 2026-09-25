"""Three original-source full-prefix bowl-row1 trajectories, one current row only."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig
from transformers.integrations import sdpa_attention
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.model_profiles.mature_source import load_saved_source as _source
from probes.recurrence_dynamics.coordinate_continuity.runtime import tensor_hash
from src.qwen.native_row_scores import compare_saved_trace as _trace_compare
from src.qwen.saved_prefix import prefix_tokens as _prefix_tokens
from probes.recurrence_dynamics.numerical_feedback.select import token_hash
from probes.recurrence_dynamics.recurrence_first_arrivals.prepare import MATURE
from probes.recurrence_dynamics.recurrence_native_history_read import run as prior
from probes.model_profiles.mature_tied_untied import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity
from src.qwen.native import exact_history_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-24-recurrence-native-row-completion"
ADMISSION = UNIT / "lead-row1-admission-v1.json"
PROTOCOL = UNIT / "unit.md"
PREDICTION = UNIT / "lead-row1-prediction-v1.md"
TIME_UPDATE = UNIT / "lead-time-budget-update-v1.json"
ACCEPTANCE = UNIT / "lead-first-case-acceptance-v1.json"
PREFLIGHT = UNIT / "supporting/row1-preflight-v1.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-native-row-completion/row1-v1")
SHAS = {PROTOCOL: "260cdfa724fa0b65c2786c1b49d3021b1412328678d1bca645b5b6708bc316b3",
        ADMISSION: "0796039847c1ed0532ff3b28ff512b3244802a61129bd09dae8fb80f5d979c21",
        PREDICTION: "1e8089f0e5689b76041bc1d72fb949fd9b2973b64cd576ef8010834d81025c66",
        TIME_UPDATE: "d986f103d046b562b809389b23f291e76f84e3190ef57788a2d88f6beda84203",
        ACCEPTANCE: "c07f9f58aa92b3dcc6eddea6b5d36b8774a02cd5658d1d5aefd3918fed3a01d7"}
ARMS = ("native", "identity-mask-sham", "latest-row-mask")
TARGET, OFFSET, ROW_START, KEY = 3, 15, 10, (0, 10)
NATIVE = [151675, 151867, 151887, 152077, 151649]
PREDICTED = [151670, 151827, 151947, 152174, 151649]
TOL = 2e-4


def require(ok, why):
    if not ok:
        raise ValueError(why)


def bind(path):
    return prior.bind(Path(path))


def write_new(path, value):
    return prior.write_new(Path(path), value)


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, f"frozen binding changed: {path}")
    a = json.loads(ADMISSION.read_text())
    update = json.loads(TIME_UPDATE.read_text())
    accepted = json.loads(ACCEPTANCE.read_text())
    require(a["status"] == "lead-admitted-fixed-row1-only" and
            a["raw_output_root"] == str(OUT) and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            a["prior_sequence_gpu_hours"] == accepted["cumulative_sequence_gpu_hours"] ==
            0.2625901622919664 and
            update["status"] == "lead-recorded-user-time-budget-update" and
            update["row1_admission"]["sha256"] == SHAS[ADMISSION] and
            update["decision"].startswith("Elapsed wall/GPU time"),
            "row1 admission/time-budget/accepted-cost changed")
    c = a["case"]
    require(c["original_prefix_end"] == OFFSET and c["current_row_start"] == ROW_START and
            c["latest_row_key_span"] == list(KEY) and c["native_row_tail"] == NATIVE and
            c["trajectory_order"] == list(ARMS) and c["max_full_prefix_width"] == 1381 and
            c["max_model_forwards"] == c["max_vision_forwards"] ==
            c["max_emitted_target_tokens"] == 15 and
            a["secondary_exact_tail_prediction"] == PREDICTED and
            a["secondary_novel_prediction_slots"] == ["y1", "x2", "y2", "row terminator"],
            "frozen row1 cells/prediction/geometry changed")
    for name in ("protocol", "secondary_prediction", "first_case_acceptance",
                 "first_case_cost_correction", "registry", "preserve_first_case_producer",
                 "preserve_first_case_candidate"):
        b = a[name]
        require(bind(b["path"])["sha256"] == b["sha256"], f"frozen dependency changed: {name}")
    for name in ("first_step_reference_receipt", "first_step_reference_readback"):
        b = c[name]
        require(bind(b["path"])["sha256"] == b["sha256"], f"accepted row1 reference changed: {name}")
    for b in c["first_step_reference_vectors"].values():
        require(bind(b["path"])["sha256"] == b["sha256"], "accepted row1 vector changed")
    for name, b in a["source_bindings"].items():
        require(bind(b["path"])["sha256"] == b["sha256"], f"source changed: {name}")
    registry = json.loads(Path(a["registry"]["path"]).read_text())
    image = registry["images"]["mature:313465:0"]
    require(image["source_bindings"] == a["source_bindings"] and
            image["request_ids"] == a["full_batch_request_ids"] and
            image["full_batch_target_lengths"] == a["full_batch_native_lengths"] and
            image["group"] == "fresh-18" and image["batch_index"] == TARGET,
            "registry/source batch changed")
    return a, image


def source(q, image, device):
    raw = json.loads(Path(image["source_bindings"]["raw"]["path"]).read_text())["rows"]
    b = image["source_bindings"]
    boundary = {"group": "fresh-18", "batch_index": TARGET, "image_id": 313465,
                "raw_path": b["raw"]["path"], "trace_path": b["trace"]["path"],
                "receipt_path": b["runtime_receipt"]["path"],
                "native_tokens": raw[TARGET]["token_ids"],
                "native_token_hash": token_hash(raw[TARGET]["token_ids"])}
    panel = json.loads((MATURE / "panel.json").read_text())
    batch, raw, trace, group, planning = _source(boundary, "untied", panel, q, device)
    receipt = json.loads(Path(b["runtime_receipt"]["path"]).read_text())
    require(len(raw) == len(group["cases"]) == 4 and
            [len(x["token_ids"]) for x in raw] == [286, 109, 11, 138] and
            input_identity(batch) == receipt["input_identity"] and
            list(batch.request_ids) == image["request_ids"] and
            raw[TARGET]["token_ids"][ROW_START:ROW_START+10] ==
            [151646, 65, 9605, 151647, 151648] + NATIVE,
            "original full batch/source row changed")
    return batch, raw, trace, receipt, planning


def step_inputs(model, batch, raw, pad, emitted, *, previous_choice=None):
    t = len(emitted)
    require(0 <= t < 5 and (t == 0 or emitted[-1] == previous_choice),
            "own previous greedy token was not consumed")
    tails = _prefix_tokens(raw, OFFSET + t, pad)
    tails[TARGET] = raw[TARGET]["token_ids"][:OFFSET] + list(emitted)
    histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
    full = exact_history_inputs(model, batch.inputs, histories, pad_token_id=pad, logits_to_keep=1)
    width = int(full["input_ids"].shape[1])
    full["cache_position"] = torch.arange(width, device=full["input_ids"].device)
    prompt = width - (OFFSET + t)
    require((full["input_ids"].shape == full["attention_mask"].shape == (4, 1377+t)) and
            full["position_ids"].shape == (3, 4, 1377+t) and prompt == 1362 and
            full["input_ids"][:, prompt:].tolist() == tails and
            full["attention_mask"][:, prompt:].tolist() == [[1]*(OFFSET+t)]*4 and
            tails[2][11:] == [pad]*(4+t) and
            tails[TARGET][ROW_START:OFFSET+t] == [151646, 65, 9605, 151647, 151648]+list(emitted),
            "step full-batch tokens/EOS/pad/positions changed")
    return full, prompt


def mask_for(full, prompt, arm, t, *, target=TARGET, query_end=None, key=KEY):
    source = full["attention_mask"]
    if query_end is None:
        query_end = OFFSET+t
    require(arm in ARMS and target == TARGET and key == KEY and
            query_end == OFFSET+t and prompt == 1362 and
            source.shape == (4, 1377+t) and 0 <= t < 5,
            "wrong progressive mask target/query/key/source shape")
    base = prior.native_4d(source)
    rect = (target, 0, slice(prompt+ROW_START, prompt+query_end),
            slice(prompt+key[0], prompt+key[1]))
    require(bool(base[rect].all()), "selected historical keys not natively readable")
    if arm == "native":
        return source, base, rect
    actual = base.clone()
    if arm == "latest-row-mask":
        actual[rect] = False
    verify_mask(base, actual, rect, arm)
    return actual, base, rect


def verify_mask(base, actual, rect, arm):
    expected = base.clone()
    if arm == "latest-row-mask":
        expected[rect] = False
    require(torch.equal(actual, expected), "selected/complement/historical/companion mask changed")


def forward_step(model, full, prompt, arm, t, active):
    mask, base, rect = mask_for(full, prompt, arm, t)
    active["expected_mask"] = base if arm == "native" else mask
    active["rect"] = rect
    return model(**{**full, "attention_mask": mask})


def record_check(full, prompt, arm, t, actual_hashes, chosen, logits, selected, complement,
                 *, previous_choice=None, emitted=(), target=TARGET, key=KEY, query_end=None):
    require(target == TARGET and key == KEY and
            (query_end is None or query_end == OFFSET+t) and
            len(emitted) == t and (t == 0 or emitted[-1] == previous_choice),
            "post-forward step/greedy/rectangle changed")
    require(full["input_ids"][TARGET, prompt+OFFSET:prompt+OFFSET+t].tolist() == list(emitted),
            "post-forward consumed target tokens changed")
    expected, base, rect = mask_for(full, prompt, arm, t, target=target, key=key, query_end=query_end)
    seen = base if arm == "native" else expected
    expected_hashes = {k: tensor_hash(expected if k == "attention_mask" else full[k])
                       for k in ("input_ids", "attention_mask", "position_ids", "cache_position")}
    require(actual_hashes == expected_hashes, "post-forward consumed input/mask changed")
    other = seen.clone()
    other[rect] = False
    require(selected == int(seen[rect].sum()) and complement == tensor_hash(other),
            "post-forward same-rectangle selected/complement changed")
    require(chosen == int(torch.argmax(logits[TARGET]).item()),
            "post-forward greedy token changed")
    return {"input_hashes": expected_hashes, "selected_true_count": selected,
            "complement_sha256": complement}


def stop_kind(emitted):
    token = emitted[-1]
    slot = len(emitted)-1
    if token == 151645:
        return "eos"
    if token == 151649:
        return "complete" if slot == 4 and all(151670 <= x < 152670 for x in emitted[:4]) else "early_row_terminator"
    if slot < 4 and not 151670 <= token < 152670:
        return "malformed_coordinate_slot"
    if slot == 4:
        return "cap_missing_terminator"
    return None


def cpu_checks(batch, raw, pad):
    model = prior.ConfigOnlyRope()
    checks = []
    prior_full = None
    for t in (0, 4):
        emitted = NATIVE[:t]
        full, prompt = step_inputs(model, batch, raw, pad, emitted,
                                   previous_choice=None if t == 0 else emitted[-1])
        if t == 0:
            prior_full = full
            config = AutoConfig.from_pretrained(BASE, local_files_only=True).text_config
            config._attn_implementation = "sdpa"
            ref = create_causal_mask(config, torch.empty((*full["attention_mask"].shape, 1)),
                                     full["attention_mask"], torch.arange(full["input_ids"].shape[1]),
                                     None, position_ids=full["position_ids"][0])
            require(torch.equal(prior.native_4d(full["attention_mask"]), ref),
                    "native 4D SDPA mask differs from installed consumer")
        class Fake:
            seen = None
            def __call__(self, **kwargs):
                self.seen = {k: tensor_hash(kwargs[k]) for k in
                             ("input_ids", "attention_mask", "position_ids", "cache_position")}
                v = torch.zeros((4, 152800))
                v[TARGET, NATIVE[t]] = 1
                return SimpleNamespace(logits=v[:, None, :])
        fake = Fake()
        active = {}
        out = forward_step(fake, full, prompt, "latest-row-mask", t, active)
        logits = out.logits[:, -1]
        chosen = int(torch.argmax(logits[TARGET]).item())
        base = prior.native_4d(full["attention_mask"])
        rect = active["rect"]
        seen = active["expected_mask"]
        complement = seen.clone(); complement[rect] = False
        selected = int(seen[rect].sum())
        record_check(full, prompt, "latest-row-mask", t, fake.seen, chosen, logits, selected,
                     tensor_hash(complement), previous_choice=None if t == 0 else emitted[-1],
                     emitted=emitted)
        checks.append(f"actual_caller_and_post_receipt_step{t}")
        for label, changes in (
            ("fixed_query_extent", {"query_end": OFFSET}),
            ("wrong_key", {"key": (10, 20)}),
            ("wrong_target", {"target": 2}),
            ("wrong_greedy", {"chosen": NATIVE[t]+1}),
            ("wrong_previous_greedy", {"previous_choice": -1}),
            ("companion_changed", {"actual_hashes": {**fake.seen, "input_ids": "0"*64}}),
        ):
            if t == 0 and label in ("fixed_query_extent", "wrong_previous_greedy"):
                continue
            args = {"full": full, "prompt": prompt, "arm": "latest-row-mask", "t": t,
                    "actual_hashes": fake.seen, "chosen": chosen, "logits": logits,
                    "selected": selected, "complement": tensor_hash(complement),
                    "previous_choice": None if t == 0 else emitted[-1], "emitted": emitted}
            args.update(changes)
            try:
                record_check(**args)
            except ValueError:
                checks.append(f"reject_{label}_step{t}")
            else:
                raise AssertionError(f"post-forward receipt accepted {label} at step {t}")
        for label, idx in (("historical", (TARGET, prompt+ROW_START-1)),
                           ("companion", (0, prompt+ROW_START))):
            bad = {**full, "input_ids": full["input_ids"].clone()}
            bad["input_ids"][idx] += 1
            try:
                record_check(bad, prompt, "latest-row-mask", t, fake.seen, chosen, logits, selected,
                             tensor_hash(complement), previous_choice=None if t == 0 else emitted[-1],
                             emitted=emitted)
            except ValueError:
                checks.append(f"reject_{label}_input_step{t}")
            else:
                raise AssertionError(f"post-forward receipt accepted changed {label}")
    require(stop_kind(NATIVE) == "complete" and
            stop_kind([151649]) == "early_row_terminator" and
            stop_kind([151645]) == "eos" and
            stop_kind([152670]) == "malformed_coordinate_slot" and
            stop_kind([152669]) is None and
            stop_kind([152670,151670,151670,151670,151649]) == "early_row_terminator" and
            stop_kind([151670,151670,151670,151670,152670]) == "cap_missing_terminator" and
            stop_kind([151670,151670,151670,151670,151670]) == "cap_missing_terminator",
            "stop parser boundaries changed")
    require(prior_full is not None, "missing CPU step0")
    return checks


def preflight():
    a, image = contract()
    require(not PREFLIGHT.exists() and not OUT.exists(), "row1 attempt already staged or run")
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU preflight loaded language model")
    batch, raw, _trace, receipt, planning = source(q, image, torch.device("cpu"))
    pad = int(q.tokenizer.pad_token_id)
    checks = cpu_checks(batch, raw, pad)
    shape = []
    hashes = []
    for t in range(5):
        full, prompt = step_inputs(prior.ConfigOnlyRope(), batch, raw, pad, NATIVE[:t],
                                   previous_choice=None if t == 0 else NATIVE[t-1])
        shape.append({"step": t, "width": int(full["input_ids"].shape[1]), "prompt": prompt,
                      "query": [prompt+ROW_START, prompt+OFFSET+t],
                      "key": [prompt+KEY[0], prompt+KEY[1]]})
        hashes.append({k: tensor_hash(full[k]) for k in
                       ("input_ids", "attention_mask", "position_ids", "cache_position")})
    require([s["width"] for s in shape] == [1377, 1378, 1379, 1380, 1381] and
            int(batch.inputs["pixel_values"].numel()) == 24403968,
            "maximum full-prefix/vision shape changed")
    forecast = 2*a["prior_unit_charge_gpu_seconds"]*(1381/1391)**2
    require(abs(forecast-a["planning_forecast_gpu_seconds"]) < 1e-6 and
            int(batch.inputs["pixel_values"].numel()) == 24403968,
            "source-shape forecast changed")
    vocab = AutoConfig.from_pretrained(BASE, local_files_only=True).text_config.vocab_size
    hidden = AutoConfig.from_pretrained(BASE, local_files_only=True).text_config.hidden_size
    artifact_forecast = 15*(4*vocab*4 + 28*(10+3)*hidden*4) + 16*1024**2
    require(artifact_forecast < a["artifact_plan_bytes"], "artifact plan over cap")
    direct = [Path(__file__), Path(prior.__file__), Path(inspect.getfile(_source)),
              Path(inspect.getfile(_prefix_tokens)), Path(inspect.getfile(token_hash)),
              Path(inspect.getfile(_trace_compare)), Path(__file__).resolve().parents[3] / 'probes/recurrence_dynamics/recurrence_first_arrivals/prepare.py',
              Path(inspect.getfile(load_model)), Path(inspect.getfile(exact_history_inputs)),
              Path(inspect.getfile(input_identity)), Path(inspect.getfile(bind)),
              Path(inspect.getfile(preserve_source)), Path(inspect.getfile(load_qwen_components_from_options)),
              Path(inspect.getfile(modeling_qwen3_vl)), Path(inspect.getfile(create_causal_mask)),
              Path(inspect.getfile(sdpa_attention))]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else Path("transformers")/path.name
        saved = preserve_source(path, run_root=OUT, relative_name=rel)
        captures.append({"maintained": bind(path), "capture": bind(saved)})
    packet = {"schema": "recurrence_native_row_completion.row1_preflight.v1",
              "status": "cpu_qualified_before_gpu", "admission": bind(ADMISSION),
              "protocol": bind(PROTOCOL), "prediction": bind(PREDICTION),
              "time_update": bind(TIME_UPDATE), "producer": bind(Path(__file__)),
              "source_bindings": image["source_bindings"], "source_identity": receipt["identity"],
              "input_identity": receipt["input_identity"], "planning": planning,
              "shape": shape, "native_source_hashes": hashes,
              "pixel_elements": int(batch.inputs["pixel_values"].numel()),
              "cpu_actual_caller_checks": checks, "forecast_outer_gpu_seconds": forecast,
              "artifact_forecast_bytes": artifact_forecast,
              "direct_source_captures": captures,
              "commands": {"gpu": ["python", "-B", "-m",
                                   "probes.recurrence_dynamics.recurrence_native_row_completion.row1", "run"],
                           "readback": ["python", "-B", "-m",
                                        "probes.recurrence_dynamics.recurrence_native_row_completion.row1", "readback"]}}
    write_new(PREFLIGHT, packet)
    print(json.dumps({"status": packet["status"], "shapes": shape,
                      "forecast_seconds": forecast, "artifact_bytes": artifact_forecast,
                      "checks": checks, "captures": len(captures)}))


def checked_preflight():
    a, image = contract()
    p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_gpu" and p["admission"] == bind(ADMISSION)
            and p["protocol"] == bind(PROTOCOL) and p["prediction"] == bind(PREDICTION) and
            p["time_update"] == bind(TIME_UPDATE) and p["producer"] == bind(Path(__file__)),
            "preflight/producer changed")
    for c in p["direct_source_captures"]:
        require(bind(c["maintained"]["path"]) == c["maintained"] and
                bind(c["capture"]["path"]) == c["capture"], "direct source capture changed")
    return a, image, p


def run():
    a, image, p = checked_preflight()
    require(not (OUT/"launch.json").exists() and
            (not OUT.exists() or {x.name for x in OUT.iterdir()} == {"row1-gpu-stdout-v1.log"}),
            "attempt exists; no retry")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    counts = {"model_forwards": 0, "vision_forwards": 0, "emitted_target_tokens": 0}
    receipt = {"schema": "recurrence_native_row_completion.row1_attempt.v1",
               "status": "running", "pid": os.getpid(), "begun_unix": time.time(),
               "admission": bind(ADMISSION), "preflight": bind(PREFLIGHT),
               "producer": bind(Path(__file__)), "counts": counts, "arms": []}
    write_new(OUT/"launch.json", receipt)
    device = torch.device("cuda:0")
    handles = []
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = load_model("untied", device)
        expected = p["source_identity"]
        require({k:v for k,v in identity.items() if k != "loader_source"} ==
                {k:v for k,v in expected.items() if k != "loader_source"} and
                all(identity["loader_source"][k] == expected["loader_source"][k]
                    for k in ("sha256", "size_bytes")), "effective model identity changed")
        model = q.model.eval()
        batch, raw, trace, source_receipt, planning = source(q, image, device)
        require(source_receipt["input_identity"] == p["input_identity"] and
                planning == p["planning"], "source preparation differs from CPU")
        layers = [m for m in model.modules() if isinstance(m, modeling_qwen3_vl.Qwen3VLTextDecoderLayer)]
        attentions = [m for m in model.modules() if isinstance(m, modeling_qwen3_vl.Qwen3VLTextAttention)]
        require(len(layers) == len(attentions) == 28 and
                [m.self_attn for m in layers] == attentions, "actual 28-layer SDPA route changed")
        active = {"full": None, "prompt": None, "expected_mask": None,
                  "seen": [], "history": [], "companions": [], "actual_input": None}
        def top_hook(_module, _args, kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 15, "model-forward cap")
            full = active["full"]
            require(torch.equal(kwargs["input_ids"], full["input_ids"]) and
                    torch.equal(kwargs["position_ids"], full["position_ids"]) and
                    torch.equal(kwargs["cache_position"], full["cache_position"]),
                    "actual source/own tokens/positions changed")
            active["actual_input"] = {k: tensor_hash(kwargs[k]) for k in
                                      ("input_ids", "attention_mask", "position_ids", "cache_position")}
        def vision_hook(_module, _args):
            counts["vision_forwards"] += 1
            require(counts["vision_forwards"] <= 15, "vision cap")
        handles.extend((model.register_forward_pre_hook(top_hook, with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(vision_hook)))
        for i, attn in enumerate(attentions):
            def attn_hook(_module, _args, kwargs, layer=i):
                mask = kwargs.get("attention_mask")
                require(isinstance(mask, torch.Tensor) and mask.ndim == 4 and
                        torch.equal(mask, active["expected_mask"]),
                        f"text layer {layer} consumed wrong progressive mask")
                active["seen"].append(layer)
            handles.append(attn.register_forward_pre_hook(attn_hook, with_kwargs=True))
        for i, layer in enumerate(layers):
            def layer_hook(_module, _args, output, layer_idx=i):
                states = output[0] if isinstance(output, tuple) else output
                require(isinstance(states, torch.Tensor) and states.ndim == 3,
                        "historical/companion states unavailable")
                prompt = active["prompt"]
                active["history"].append(states[TARGET, prompt:prompt+ROW_START].detach().cpu().float())
                active["companions"].append(states[[0,1,2], -1].detach().cpu().float())
            handles.append(layer.register_forward_hook(layer_hook))
        receipt["effective_identity"] = identity
        pad = int(q.tokenizer.pad_token_id)
        with torch.inference_mode():
            for arm in ARMS:
                emitted = []
                arm_record = {"arm": arm, "steps": [], "stop": None}
                receipt["arms"].append(arm_record)
                for t in range(5):
                    full, prompt = step_inputs(model, batch, raw, pad, emitted,
                                               previous_choice=None if t == 0 else arm_record["steps"][-1]["chosen"])
                    if arm in ARMS[:2]:
                        require(all(tensor_hash(full[k]) == p["native_source_hashes"][t][k]
                                    for k in ("input_ids", "attention_mask", "position_ids", "cache_position")),
                                "native/sham full-prefix source differs from preflight")
                    active.update(full=full, prompt=prompt, expected_mask=None, seen=[],
                                  history=[], companions=[], actual_input=None)
                    out = forward_step(model, full, prompt, arm, t, active)
                    torch.cuda.synchronize(device)
                    require(active["seen"] == list(range(28)) and
                            len(active["history"]) == len(active["companions"]) == 28 and
                            active["actual_input"] is not None, "actual consumer/state incomplete")
                    logits = out.logits[:, -1, :].detach().cpu().float()
                    require(logits.shape[0] == 4 and torch.isfinite(logits).all().item(),
                            "invalid full-vocabulary vectors")
                    payload = {"arm": arm, "step": t, "logits": logits,
                               "prior_target_by_layer": torch.stack(active["history"]),
                               "companion_last_by_layer": torch.stack(active["companions"])}
                    raw_path = OUT/f"{arm}-step{t}.pt"
                    torch.save(payload, raw_path)
                    chosen = int(torch.argmax(logits[TARGET]).item())
                    mask, base, rect = mask_for(full, prompt, arm, t)
                    seen = base if arm == "native" else mask
                    complement = seen.clone(); complement[rect] = False
                    selected = int(seen[rect].sum())
                    meta = record_check(full, prompt, arm, t, active["actual_input"], chosen, logits,
                                        selected, tensor_hash(complement),
                                        previous_choice=None if t == 0 else arm_record["steps"][-1]["chosen"],
                                        emitted=emitted)
                    entry = {"step": t, "raw": bind(raw_path), "input_hashes": meta["input_hashes"],
                             "actual_mask_layers": active["seen"], "query_physical": [prompt+ROW_START,prompt+OFFSET+t],
                             "key_physical": [prompt+KEY[0],prompt+KEY[1]],
                             "selected_true_count": selected, "complement_sha256": meta["complement_sha256"],
                             "chosen": chosen, "elapsed_gpu_seconds": time.monotonic()-started}
                    write_new(OUT/f"inputs-{arm}-step{t}.json",
                              {k: full[k].detach().cpu().tolist() for k in
                               ("input_ids", "attention_mask", "position_ids", "cache_position")})
                    entry["inputs"] = bind(OUT/f"inputs-{arm}-step{t}.json")
                    if t == 0:
                        ref = a["case"]["first_step_reference_vectors"][arm]
                        require(bind(ref["path"])["sha256"] == ref["sha256"], "accepted step0 vector changed")
                        old = torch.load(ref["path"], map_location="cpu", weights_only=True)["logits"]
                        entry["accepted_step0_full_vector_max_error"] = float((logits-old).abs().max())
                        require(entry["accepted_step0_full_vector_max_error"] <= TOL,
                                f"{arm} accepted first-step full vector mismatch")
                    if arm == "native":
                        active_rows = [i for i,r in enumerate(raw) if len(r["token_ids"]) > OFFSET+t]
                        entry["source_trace_parity"] = [_trace_compare(
                            logits=logits[i], trace=trace, batch_index=i,
                            absolute_offset=OFFSET+t, token_id=int(raw[i]["token_ids"][OFFSET+t]),
                            role="native_row_completion") for i in active_rows]
                        require(active_rows == [0,1,3] and
                                all(x["passed"] for x in entry["source_trace_parity"]) and
                                chosen == NATIVE[t], f"native source/greedy parity failed at step {t}")
                    else:
                        native = torch.load(OUT/f"native-step{t}.pt", map_location="cpu", weights_only=True)
                        entry["prior_history_max_error"] = float((payload["prior_target_by_layer"]-
                                                                  native["prior_target_by_layer"]).abs().max())
                        entry["companion_hidden_max_error"] = float((payload["companion_last_by_layer"]-
                                                                    native["companion_last_by_layer"]).abs().max())
                        require(max(entry["prior_history_max_error"],
                                    entry["companion_hidden_max_error"]) <= TOL,
                                "historical/companion layer states changed")
                        if arm == "identity-mask-sham":
                            entry["all_batch_vector_max_error"] = float((logits-native["logits"]).abs().max())
                            require(entry["all_batch_vector_max_error"] <= TOL and chosen == NATIVE[t],
                                    "sham source/full-vector/greedy mismatch")
                        else:
                            entry["companion_vector_max_errors"] = [
                                float((logits[i]-native["logits"][i]).abs().max()) for i in (0,1,2)]
                            native_step = receipt["arms"][0]["steps"][t]
                            require(max(entry["companion_vector_max_errors"]) <= TOL and
                                    selected == 0 and native_step["selected_true_count"] == (5+t)*10 and
                                    entry["complement_sha256"] == native_step["complement_sha256"],
                                    "treatment companion/same-rectangle complement changed")
                    arm_record["steps"].append(entry)
                    emitted.append(chosen)
                    counts["emitted_target_tokens"] += 1
                    require(counts["emitted_target_tokens"] <= 15, "emitted-token cap")
                    arm_record["emitted"] = list(emitted)
                    arm_record["stop"] = stop_kind(emitted)
                    receipt["counts"] = dict(counts)
                    receipt["allocated_gpu_seconds"] = time.monotonic()-started
                    write_new(OUT/f"checkpoint-{arm}-{t}.json", receipt)
                    if arm_record["stop"] is not None:
                        break
                require(arm_record["stop"] is not None, "trajectory did not stop at five-token cap")
                if arm in ARMS[:2]:
                    require(arm_record["emitted"] == NATIVE and arm_record["stop"] == "complete",
                            "native/sham failed original complete row")
        receipt["status"] = "candidate_complete"
    except BaseException as exc:
        receipt["status"] = "technical_invalid"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()}
    finally:
        for h in handles:
            h.remove()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        receipt["allocated_gpu_seconds"] = time.monotonic()-started
        receipt["wall_seconds"] = receipt["allocated_gpu_seconds"]
        receipt["counts"] = dict(counts)
        receipt["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        receipt["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
        receipt["terminal_pid"] = os.getpid()
        receipt["artifact_bytes"] = sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
        write_new(OUT/"receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "counts": counts,
                      "gpu_seconds": receipt["allocated_gpu_seconds"],
                      "failure": receipt.get("failure", {}).get("message")}))
    require(receipt["status"] == "candidate_complete", "row1 attempt failed; no retry")


def readback():
    a, image, p = checked_preflight()
    receipt = json.loads((OUT/"receipt.json").read_text())
    require(receipt["status"] == "candidate_complete" and receipt["admission"] == bind(ADMISSION) and
            len(receipt["arms"]) == 3 and [x["arm"] for x in receipt["arms"]] == list(ARMS) and
            receipt["counts"]["model_forwards"] == receipt["counts"]["vision_forwards"] ==
            receipt["counts"]["emitted_target_tokens"] <= 15,
            "attempt/order/count changed")
    require(not (OUT/"readback.json").exists(), "cold readback exists")
    raw = json.loads(Path(image["source_bindings"]["raw"]["path"]).read_text())["rows"]
    trace = json.loads(Path(image["source_bindings"]["trace"]["path"]).read_text())
    summary = {"schema": "recurrence_native_row_completion.row1_readback.v1",
               "status": "candidate_cold_readback_passed", "receipt": bind(OUT/"receipt.json"),
               "admission": bind(ADMISSION), "arms": [], "counts": receipt["counts"]}
    all_vectors = {}
    for arm_rec in receipt["arms"]:
        arm = arm_rec["arm"]
        vectors = []
        emitted = []
        summaries = []
        for t, entry in enumerate(arm_rec["steps"]):
            require(entry["step"] == t and entry["actual_mask_layers"] == list(range(28)) and
                    bind(entry["raw"]["path"]) == entry["raw"] and
                    bind(entry["inputs"]["path"]) == entry["inputs"],
                    "raw/input/all-layer evidence changed")
            values = json.loads(Path(entry["inputs"]["path"]).read_text())
            full = {k: torch.tensor(values[k], dtype=torch.long) for k in
                    ("input_ids", "attention_mask", "position_ids", "cache_position")}
            payload = torch.load(entry["raw"]["path"], map_location="cpu", weights_only=True)
            require(payload["arm"] == arm and payload["step"] == t and
                    tuple(payload["logits"].shape)[0] == 4 and torch.isfinite(payload["logits"]).all().item() and
                    payload["prior_target_by_layer"].shape[0] == 28 and
                    payload["companion_last_by_layer"].shape[:2] == (28,3),
                    "raw vectors/states incomplete")
            logits = payload["logits"]
            mask, base, rect = mask_for(full, 1362, arm, t)
            seen = base if arm == "native" else mask
            complement = seen.clone(); complement[rect] = False
            record_check(full, 1362, arm, t, entry["input_hashes"], entry["chosen"], logits,
                         entry["selected_true_count"], entry["complement_sha256"],
                         previous_choice=None if t == 0 else emitted[-1], emitted=emitted)
            require(entry["complement_sha256"] == tensor_hash(complement) and
                    entry["query_physical"] == [1372,1377+t] and entry["key_physical"] == [1362,1372],
                    "same-rectangle/progressive geometry changed")
            if arm in ARMS[:2]:
                require({k:tensor_hash(full[k]) for k in p["native_source_hashes"][t]} ==
                        p["native_source_hashes"][t], "source native/sham inputs changed")
            if t == 0:
                ref = a["case"]["first_step_reference_vectors"][arm]
                require(bind(ref["path"])["sha256"] == ref["sha256"] and
                        float((logits-torch.load(ref["path"],map_location="cpu",weights_only=True)["logits"]).abs().max())
                        <= TOL, "accepted step0 full vector changed")
            if arm == "native":
                parity = [_trace_compare(logits=logits[i], trace=trace, batch_index=i,
                                         absolute_offset=OFFSET+t,
                                         token_id=raw[i]["token_ids"][OFFSET+t],
                                         role="cold_native_row_completion") for i in (0,1,3)]
                require(all(x["passed"] for x in parity) and entry["source_trace_parity"] and
                        entry["chosen"] == NATIVE[t], "cold source trace parity failed")
            else:
                native = all_vectors["native"][t]
                require(float((payload["prior_target_by_layer"]-
                               native["prior_target_by_layer"]).abs().max()) <= TOL and
                        float((payload["companion_last_by_layer"]-
                               native["companion_last_by_layer"]).abs().max()) <= TOL,
                        "cold prior/companion states changed")
                if arm == "identity-mask-sham":
                    require(float((logits-native["logits"]).abs().max()) <= TOL and
                            entry["chosen"] == NATIVE[t], "cold sham vectors/greedy changed")
                else:
                    require(all(float((logits[i]-native["logits"][i]).abs().max()) <= TOL for i in (0,1,2)),
                            "cold treatment companion vectors changed")
                    native_entry = receipt["arms"][0]["steps"][t]
                    require(entry["complement_sha256"] == native_entry["complement_sha256"] and
                            entry["selected_true_count"] == 0 and
                            native_entry["selected_true_count"] == (5+t)*10,
                            "cold same-step/same-rectangle mask complement changed")
            v = logits[TARGET].double()
            top = torch.topk(v,2)
            probs = torch.softmax(v,-1)
            row = {"step":t, "chosen":entry["chosen"], "top2_ids":top.indices.tolist(),
                   "top2_logits":top.values.tolist(), "gap":float(top.values[0]-top.values[1]),
                   "chosen_prob":float(probs[entry["chosen"]]),
                   "chosen_logprob":float(torch.log_softmax(v,-1)[entry["chosen"]]),
                   "raw":entry["raw"]}
            summaries.append(row)
            vectors.append(payload)
            emitted.append(entry["chosen"])
            require(stop_kind(emitted) is None if t < len(arm_rec["steps"])-1
                    else stop_kind(emitted) == arm_rec["stop"], "cold stop/parser changed")
        require(emitted == arm_rec["emitted"], "cold own generated history changed")
        all_vectors[arm] = vectors
        summary["arms"].append({"arm":arm, "emitted":emitted, "stop":arm_rec["stop"],
                                "steps":summaries})
    native = summary["arms"][0]["emitted"]
    treated = summary["arms"][2]["emitted"]
    summary["first_divergence"] = next((i for i,(x,y) in enumerate(zip(native,treated)) if x != y),
                                       None if len(native)==len(treated) else min(len(native),len(treated)))
    summary["prior_sequence_gpu_hours"] = a["prior_sequence_gpu_hours"]
    summary["case_gpu_seconds"] = receipt["allocated_gpu_seconds"]
    summary["sequence_gpu_hours"] = a["prior_sequence_gpu_hours"]+receipt["allocated_gpu_seconds"]/3600
    require(receipt["artifact_bytes"] < a["artifact_plan_bytes"],
            "cold artifact bound failed")
    summary["secondary_prediction"] = {
        "frozen_tail": PREDICTED,
        "x1_was_previously_observed": True,
        "novel_slots": a["secondary_novel_prediction_slots"],
        "exact_remaining_tail_match": treated[1:] == PREDICTED[1:] if len(treated) == 5 else False,
        "full_tail_match": treated == PREDICTED,
        "category": "supported" if treated == PREDICTED else "scientific_negative_exact_miss",
    }
    write_new(OUT/"readback.json", summary)
    print(json.dumps({"status":summary["status"], "rows":{x["arm"]:x["emitted"] for x in summary["arms"]},
                      "stops":{x["arm"]:x["stop"] for x in summary["arms"]},
                      "first_divergence":summary["first_divergence"],
                      "internal_gpu_seconds":summary["case_gpu_seconds"],
                      "secondary":summary["secondary_prediction"]["category"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight","run","readback"))
    {"preflight":preflight,"run":run,"readback":readback}[parser.parse_args().action]()
