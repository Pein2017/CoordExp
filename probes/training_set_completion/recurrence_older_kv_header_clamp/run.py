"""One admitted 22-call AF/FF current-header clamp, with cold readback."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.recurrence_older_kv_split import run as old
from probes.training_set_completion.recurrence_key_phase import rotate
from src.artifacts.source_provenance import preserve_source


ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "research/experiments/2026-09-24-recurrence-older-kv-header-clamp"
PROTOCOL, ADMISSION = UNIT / "unit.md", UNIT / "lead-admission-v1.json"
PREFLIGHT = UNIT / "supporting/attempt-001-preflight.json"
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-kv-header-clamp/attempt-001")
SHAS = {PROTOCOL: "3d2ad946c4fe5560987fa14c4f9de364c0d7e8c7e0abe2c2a0a3f5b3418c1cbe",
        ADMISSION: "4cbaa75a831698324550c4b3d2dbc9a9e5636fcaff5c9c595cf97e205ff5cb77"}
NAMES = old.NAMES + (
    "clamp_native_AF", "clamp_native_FF", "clamp_joint_AF_from_FF",
    "clamp_joint_FF_from_AF", "clamp_V_AF_from_FF", "clamp_V_FF_from_AF",
    "clamp_K_AF_from_FF", "clamp_K_FF_from_AF",
)
TARGET, WIDTH, END, LAYERS, KV_HEADS, Q_HEADS, DIM = 2, 1380, 1384, 28, 8, 16, 128
TOL, POSTK_TOL, HEAD_REL = 2e-4, 2e-5, 5e-5
car = old.car
bind, require, write_new = old.bind, old.require, old.write_new
cache_digest = old.cache_digest


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, "frozen header-clamp contract changed")
    a = json.loads(ADMISSION.read_text())
    require(a["status"] == "lead-admitted-twenty-two-calls-after-CPU-qualification" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            [c["name"] for c in a["cells"]] == list(NAMES) and
            [c["vision"] for c in a["cells"]] == [1] * 4 + [0] * 18 and
            a["max_model_forwards"] == 22 and a["max_vision_forwards"] == 4 and
            a["max_generated_tokens"] == 0 and a["max_model_jobs"] == 1 and
            a["output_root"] == str(OUT) and a["shape"]["target_index"] == TARGET and
            a["shape"]["older_physical"] == list(old.OLDER) and
            a["shape"]["latest_physical"] == list(old.LATEST) and
            a["shape"]["header_physical"] == [WIDTH, END] and
            a["qualification"]["max_abs_logits"] == TOL and
            a["qualification"]["appended_current_K_max_abs"] == POSTK_TOL and
            a["qualification"]["headout_relative_bound"] == HEAD_REL,
            "finite header-clamp admission changed")
    for key in ("protocol", "predecessor_acceptance", "predecessor_admission",
                "predecessor_verification", "predecessor_producer", "cpu_acceptance",
                "cpu_candidate", "cpu_bindings", "predecessor_receipt",
                "predecessor_readback", "predecessor_outer", "source_panel"):
        require(bind(a[key]["path"]) == a[key], f"bound {key} changed")
    for value in a["maintained_route"].values():
        require(bind(value["path"]) == value, "maintained route changed")
    for key in ("raw", "trace", "runtime_receipt", "image"):
        v = a["source_bindings"][key]
        require(bind(v["path"]) == v, f"source {key} changed")
    require(bind(a["loader_crosswalk"]["maintained"]["path"]) ==
            a["loader_crosswalk"]["maintained"], "maintained loader changed")
    for group in ("saved_full_vector_references", "saved_full_inputs",
                  "anchor_references"):
        for v in a[group].values():
            require(bind(v["path"]) == v, f"{group} changed")
    for group in ("adaptive_references", "joint_references"):
        for cell in a[group].values():
            for v in cell.values():
                if isinstance(v, dict) and "path" in v:
                    require(bind(v["path"]) == v, f"{group} changed")
    old_a = old.contract()
    require(bind(old.ADMISSION) == a["predecessor_admission"] and
            bind(old.PROTOCOL) == old_a["protocol"], "accepted older-split binding changed")
    return a, old_a


def axes(cell):
    return [cell[k] for k in ("base_origin", "older_K_origin", "older_V_origin", "latest_KV_origin")]


def record_guard(cells, admitted):
    require(isinstance(cells, list) and len(cells) == 22, "serialized cell container/count changed")
    for i, (cell, source) in enumerate(zip(cells, admitted, strict=True)):
        require(isinstance(cell, dict) and cell.get("call") == i + 1 and
                cell.get("name") == NAMES[i] and cell.get("origins") == axes(source) and
                cell.get("clamp") is source["clamp"] and
                isinstance(cell.get("input"), dict) and isinstance(cell.get("consumer"), dict),
                "serialized cell order/origins/container changed")


def clamp_output(output, expected, *, target=TARGET, axis="Q"):
    shapes = {"Q": (4, 4, Q_HEADS, DIM), "K": (4, 4, KV_HEADS, DIM),
              "V": (4, 4, KV_HEADS * DIM)}
    require(axis in shapes and target == TARGET and output.shape == shapes[axis] and
            expected.shape == output[target].shape and output.dtype == expected.dtype and
            output.device == expected.device, "current-header clamp target/span/axis changed")
    changed = output.clone()
    changed[target].copy_(expected)
    require(torch.equal(changed[target], expected) and
            torch.equal(changed[[0, 1, 3]], output[[0, 1, 3]]),
            "current-header clamp touched companion")
    return changed


@contextmanager
def header_hooks(model, active, captures):
    """Target-only Q/K/V output hooks; actuator precedes observer on every layer."""
    handles = []
    try:
        for i, layer in enumerate(model.model.language_model.layers):
            for axis, module in (("Q", layer.self_attn.q_norm),
                                 ("K", layer.self_attn.k_norm),
                                 ("V", layer.self_attn.v_proj)):
                def actuate(_module, _args, output, *, index=i, name=axis):
                    if active["kind"] != "suffix":
                        return output
                    if active["clamp"]:
                        require(active["base"] in captures and
                                len(captures[active["base"]]) == LAYERS and
                                name in captures[active["base"]][index],
                                "missing own-base native Q/K/V capture")
                        return clamp_output(output, captures[active["base"]][index][name], axis=name)
                    return output

                def observe(_module, _args, output, *, index=i, name=axis):
                    if active["kind"] != "suffix":
                        return
                    shapes = {"Q": (4, 4, Q_HEADS, DIM), "K": (4, 4, KV_HEADS, DIM),
                              "V": (4, 4, KV_HEADS * DIM)}
                    require(output.shape == shapes[name] and
                            name not in active["qkv"][index],
                            "current-header Q/K/V shape or repeat changed")
                    target = output[TARGET].detach()
                    companions = output[[0, 1, 3]].detach()
                    if active["clamp"]:
                        expected = captures[active["base"]][index][name]
                        require(torch.equal(target, expected),
                                "actual module output did not consume own-base clamp")
                    active["qkv"][index][name] = target
                    active["qkv_hashes"][index][name] = {
                        "target": car.base.tensor_hash(target),
                        "companions": car.base.tensor_hash(companions)}
                    if active["capture"]:
                        captures[active["base"]][index][name] = target.clone()
                handles.append(module.register_forward_hook(actuate))
                handles.append(module.register_forward_hook(observe))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def observe_headout(active, layer, index, projection_input):
    """Compare the real pre-o_proj SDPA output with an FP64 read of actual cache."""
    require(active["kind"] == "suffix" and
            set(active["qkv"][index]) == {"Q", "K", "V"} and
            len(active["rotary_values"]) == 1,
            "actual Q/K/V or rotary observation incomplete")
    cache = active["cache"]
    cos, sin = active["rotary_values"][0]
    phase_cos, phase_sin = cos[TARGET], sin[TARGET]
    q = active["qkv"][index]["Q"].transpose(0, 1)
    pre_k = active["qkv"][index]["K"].transpose(0, 1)
    pre_v = active["qkv"][index]["V"].reshape(4, KV_HEADS, DIM).transpose(0, 1)
    keys, values = cache.layers[index].keys[TARGET], cache.layers[index].values[TARGET]
    require(keys.shape == values.shape == (KV_HEADS, END, DIM) and
            cache.get_seq_length(index) == END, "actual appended header cache shape changed")
    k_error = float((rotate(pre_k, phase_cos, phase_sin) - keys[:, WIDTH:END]).abs().max())
    v_error = float((pre_v - values[:, WIDTH:END]).abs().max())
    require(k_error <= POSTK_TOL and v_error == 0,
            "current-header K/V did not reach actual attention cache")
    post_q = rotate(q, phase_cos, phase_sin)
    mask = active["expected_mask"][TARGET, 0]
    require(mask.dtype == torch.bool and mask.shape == (4, END),
            "actual target attention mask changed")
    repeated_k = keys.double().repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
    repeated_v = values.double().repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
    scores = torch.matmul(post_q.double(), repeated_k.transpose(-1, -2)) * layer.self_attn.scaling
    probs = torch.softmax(scores.masked_fill(~mask.unsqueeze(0), -torch.inf), dim=-1)
    reconstructed = torch.matmul(probs, repeated_v)
    actual = projection_input[TARGET].reshape(4, Q_HEADS, DIM).transpose(0, 1).double()
    require(torch.isfinite(reconstructed).all().item() and actual.shape == reconstructed.shape,
            "actual attention head output nonfinite/shape changed")
    error = float((actual - reconstructed).abs().max())
    scale = max(1.0, float(actual.abs().max()), float(reconstructed.abs().max()))
    bound = HEAD_REL * scale
    packet = {"layer": index, "max_abs_error": error, "scale": scale, "bound": bound,
              "postK_max_abs_error": k_error, "V_max_abs_error": v_error,
              "postQ_hash": car.base.tensor_hash(post_q),
              "appended_K_hash": car.base.tensor_hash(keys[:, WIDTH:END]),
              "appended_V_hash": car.base.tensor_hash(values[:, WIDTH:END]),
              "actual_headout_hash": car.base.tensor_hash(actual.float()),
              "reconstructed_headout_hash": car.base.tensor_hash(reconstructed)}
    active["headout_packets"].append(packet)
    active["headouts"].append({"actual": actual.float().cpu(),
                               "reconstructed": reconstructed.cpu()})
    require(error <= bound, f"layer {index} actual SDPA headout reconstruction failed: {error}>{bound}")
    return packet


def cpu_checks(a, old_a, q, batch, raw, pad):
    checks, fulls = old.cpu_checks(q, batch, raw, pad, old_a)
    require(int(batch.inputs["pixel_values"].numel()) == 24502272 and
            len(checks) >= 55, "accepted CPU source/older-patch checks incomplete")
    for origin in ("AF", "FF"):
        require({k: car.base.tensor_hash(v) for k, v in fulls[origin].items()
                 if isinstance(v, torch.Tensor)} ==
                json.loads(old.PREFLIGHT.read_text())["full_inputs"][origin],
                "current full input differs from accepted predecessor")
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.v_proj = torch.nn.Identity()
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(Layer() for _ in range(LAYERS))
    model = Model()
    fixture = {"kind": "suffix", "clamp": False, "capture": True, "base": "AF",
               "qkv": [{} for _ in range(LAYERS)], "qkv_hashes": [{} for _ in range(LAYERS)]}
    captures = {"AF": [{} for _ in range(LAYERS)], "FF": [{} for _ in range(LAYERS)]}
    original = {"Q": torch.ones(4, 4, Q_HEADS, DIM),
                "K": torch.ones(4, 4, KV_HEADS, DIM) * 2,
                "V": torch.ones(4, 4, KV_HEADS * DIM) * 3}
    with header_hooks(model, fixture, captures):
        for layer in model.model.language_model.layers:
            for axis, module in (("Q", layer.self_attn.q_norm),
                                 ("K", layer.self_attn.k_norm),
                                 ("V", layer.self_attn.v_proj)):
                module(original[axis])
    require(all(set(row) == {"Q", "K", "V"} for row in captures["AF"]),
            "CPU native capture did not cover all layers/axes")
    fixture.update(clamp=True, capture=False, qkv=[{} for _ in range(LAYERS)],
                   qkv_hashes=[{} for _ in range(LAYERS)])
    with header_hooks(model, fixture, captures):
        for layer in model.model.language_model.layers:
            for axis, module in (("Q", layer.self_attn.q_norm),
                                 ("K", layer.self_attn.k_norm),
                                 ("V", layer.self_attn.v_proj)):
                changed = original[axis] + 7
                consumed = module(changed)
                require(torch.equal(consumed[TARGET], original[axis][TARGET]) and
                        torch.equal(consumed[[0, 1, 3]], changed[[0, 1, 3]]),
                        "CPU actual module caller clamp identity/companion failed")
    require(all(set(row) == {"Q", "K", "V"} for row in fixture["qkv"]),
            "CPU all-axis output observer incomplete")
    checks.append("actual_28_layer_all_four_QKV_hook_capture_clamp_consumer")
    for label, value, expected, kw in (
            ("wrong_target", original["Q"], original["Q"][TARGET], {"target": 1, "axis": "Q"}),
            ("dropped_position", original["K"], original["K"][TARGET, :3], {"axis": "K"}),
            ("wrong_axis", original["V"], original["V"][TARGET], {"axis": "Q"}),
            ("wrong_span", original["Q"][:, :3], original["Q"][TARGET], {"axis": "Q"})):
        try:
            clamp_output(value, expected, **kw)
        except ValueError:
            checks.append("reject_" + label)
        else:
            raise AssertionError("CPU clamp accepted " + label)
    fixture["base"] = "FF"
    try:
        with header_hooks(model, fixture, captures):
            model.model.language_model.layers[0].self_attn.q_norm(original["Q"])
    except ValueError:
        checks.append("reject_missing_wrong_base_native_capture")
    else:
        raise AssertionError("CPU clamp admitted wrong base capture")
    fixture["base"] = "AF"
    try:
        with header_hooks(model, fixture, captures):
            raise RuntimeError("forced hook body failure")
    except RuntimeError as exc:
        require(str(exc) == "forced hook body failure", "wrong CPU hook failure")
    require(all(not m._forward_hooks for layer in model.model.language_model.layers
                for m in (layer.self_attn.q_norm, layer.self_attn.k_norm,
                          layer.self_attn.v_proj)), "forced exception leaked Q/K/V hooks")
    checks.append("forced_exception_all_hook_removal")
    stub = {"input": {}, "consumer": {}}
    good = [{**stub, "call": i + 1, "name": NAMES[i],
             "origins": axes(c), "clamp": c["clamp"]} for i, c in enumerate(a["cells"])]
    record_guard(json.loads(json.dumps(good)), a["cells"])
    checks.append("serialized_22_cell_guard_green")
    for label, bad in (("swapped", good[:15] + [good[16], good[15]] + good[17:]),
                       ("dropped", good[:-1]), ("extra", good + [good[-1]]),
                       ("wrong_clamp", good[:14] + [{**good[14], "clamp": False}] + good[15:]),
                       ("wrong_donor", good[:16] + [{**good[16], "origins": axes(a["cells"][16])[:1] + ["AF"] * 3}] + good[17:]),
                       ("wrong_container", {"cells": good})):
        try:
            record_guard(json.loads(json.dumps(bad)), a["cells"])
        except ValueError:
            checks.append("serialized_reject_" + label)
        else:
            raise AssertionError("serialized guard accepted " + label)
    mask = car.expected_suffix_mask(fulls["AF"])
    car.verify_attention(mask, mask)
    for label, field, index in (("header", "input_ids", (TARGET, 0)),
                                ("companion", "input_ids", (1, 0)),
                                ("position", "position_ids", (0, TARGET, 0)),
                                ("source_mask", "attention_mask", (1, WIDTH)),
                                ("cache_slot", "cache_position", (0,))):
        suffix = car.split_inputs(fulls["AF"], object(), "suffix")
        wrong = dict(suffix)
        wrong[field] = suffix[field].clone()
        wrong[field][index] += 1
        try:
            car.verify_actual_input(wrong, suffix, media=False)
        except ValueError:
            checks.append("actual_caller_reject_" + label)
        else:
            raise AssertionError("caller accepted " + label)
    wrong_mask = mask.clone()
    wrong_mask[TARGET, 0, 0, old.OLDER[0]] = ~wrong_mask[TARGET, 0, 0, old.OLDER[0]]
    try:
        car.verify_attention(wrong_mask, mask)
    except ValueError:
        checks.append("actual_attention_reject_mask")
    else:
        raise AssertionError("caller accepted wrong mask")
    # Exercise the production headout observer against real CPU SDPA, then corrupt its inputs.
    fake_cache = DynamicCache()
    keys = torch.zeros((4, KV_HEADS, END, DIM))
    values = torch.zeros_like(keys)
    keys[TARGET, :, WIDTH:END] = 1
    values[TARGET, :, WIDTH:END] = 1
    fake_cache.update(keys, values, 0)
    fake_layer = Layer()
    fake_layer.self_attn.scaling = DIM ** -0.5
    test_active = {"kind": "suffix", "cache": fake_cache,
                   "expected_mask": mask,
                   "rotary_values": [(torch.ones((4, 4, DIM)), torch.zeros((4, 4, DIM)))],
                   "qkv": [{"Q": torch.zeros((4, Q_HEADS, DIM)),
                            "K": torch.ones((4, KV_HEADS, DIM)),
                            "V": torch.ones((4, KV_HEADS * DIM))}],
                   "headouts": [], "headout_packets": []}
    head = torch.nn.functional.scaled_dot_product_attention(
        torch.zeros((Q_HEADS, 4, DIM)),
        keys[TARGET].repeat_interleave(Q_HEADS // KV_HEADS, dim=0),
        values[TARGET].repeat_interleave(Q_HEADS // KV_HEADS, dim=0),
        attn_mask=mask[TARGET, 0].unsqueeze(0))
    projected = torch.zeros((4, 4, Q_HEADS * DIM))
    projected[TARGET] = head.transpose(0, 1).reshape(4, Q_HEADS * DIM)
    observe_headout(test_active, fake_layer, 0, projected)
    checks.append("real_CPU_SDPA_headout_oracle_pass")
    for label, mutate, undo in (
            ("wrong_headout", lambda: projected[TARGET, 0, 0].add_(1),
             lambda: projected[TARGET, 0, 0].sub_(1)),
            ("wrong_postK", lambda: fake_cache.layers[0].keys[TARGET, 0, WIDTH, 0].add_(1),
             lambda: fake_cache.layers[0].keys[TARGET, 0, WIDTH, 0].sub_(1)),
            ("wrong_phase", lambda: test_active["rotary_values"][0][0][TARGET, 0, 0].zero_(),
             lambda: test_active["rotary_values"][0][0][TARGET, 0, 0].fill_(1)),
            ("wrong_consumer_mask", lambda: test_active["expected_mask"][TARGET, 0, 0, WIDTH].logical_not_(),
             lambda: test_active["expected_mask"][TARGET, 0, 0, WIDTH].logical_not_())):
        mutate()
        try:
            observe_headout(test_active, fake_layer, 0, projected)
        except ValueError:
            checks.append("headout_oracle_reject_" + label)
        else:
            raise AssertionError("headout observer accepted " + label)
        finally:
            undo()
    return checks, fulls


def preflight():
    a, old_a = contract()
    require(not PREFLIGHT.exists() and not (OUT / "launch.json").exists(),
            "attempt already prepared/launched")
    q = car.base.load_qwen_components_from_options(car.base.QwenLoadOptions(
        base_model=str(car.base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "CPU preflight loaded language model")
    batch, raw, _, sr, _ = car.prior.source(q, old_a, torch.device("cpu"))
    pad = int(q.tokenizer.pad_token_id)
    checks, fulls = cpu_checks(a, old_a, q, batch, raw, pad)
    require(sr["input_identity"] == json.loads(old.PREFLIGHT.read_text())["input_identity"] and
            list(batch.request_ids) == a["source_bindings"]["input_identity_summary"]["request_ids"],
            "current native source identity changed")
    direct = [Path(__file__), Path(old.__file__), Path(car.__file__),
              Path(car.prior.__file__), Path(car.base.__file__),
              Path(rotate.__code__.co_filename),
              Path(inspect.getfile(DynamicCache)),
              Path(inspect.getfile(modeling_qwen3_vl)),
              Path(inspect.getfile(sdpa_attention)),
              Path(preserve_source.__code__.co_filename)]
    prior_pre = json.loads(old.PREFLIGHT.read_text())
    direct += [Path(c["maintained"]["path"]) for c in prior_pre["source_captures"]]
    captures = []
    for path in dict.fromkeys(direct):
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("installed") / path.name
        saved = preserve_source(path, run_root=OUT, relative_name=rel)
        captures.append({"maintained": bind(path), "capture": bind(saved)})
    command = ["python", "-B", "-m", "probes.training_set_completion.recurrence_older_kv_header_clamp.run"]
    expected_raw = (a["measured_predecessor"]["accepted_raw_bytes"] +
                    8 * a["measured_predecessor"]["measured_suffix_cell_mean_bytes"] +
                    a["measured_predecessor"]["two_base_target_capture_bytes"] +
                    18 * 28 * 4 * Q_HEADS * DIM * (4 + 8))
    require(expected_raw < a["artifact_planning_bytes"] and
            a["planning_outer_seconds"] > 0, "selective-headout artifact forecast exceeds envelope")
    packet = {"status": "cpu_qualified_before_model", "protocol": bind(PROTOCOL),
              "admission": bind(ADMISSION), "producer": bind(Path(__file__)),
              "source_identity": sr["identity"], "input_identity": sr["input_identity"],
              "request_ids": list(batch.request_ids), "pad_id": pad,
              "pixel_elements": int(batch.inputs["pixel_values"].numel()),
              "shapes": {"full": [4, END], "prefill": [4, WIDTH], "suffix": [4, 4],
                         "suffix_mask": [4, 1, 4, END]},
              "full_inputs": {o: {k: car.base.tensor_hash(v) for k, v in full.items()
                                   if isinstance(v, torch.Tensor)} for o, full in fulls.items()},
              "checks": checks, "source_captures": captures,
              "forecast_outer_seconds": a["planning_outer_seconds"],
              "artifact_forecast_bytes_with_headouts": expected_raw,
              "artifact_envelope_bytes": a["artifact_planning_bytes"],
              "commands": {kind: command + [kind] for kind in
                           ("preflight", "run", "gpu", "readback")}}
    write_new(PREFLIGHT, packet)
    print(json.dumps({"status": packet["status"], "checks": len(checks),
                      "captures": len(captures), "artifact_forecast_bytes": expected_raw}))


def checked():
    a, old_a = contract()
    p = json.loads(PREFLIGHT.read_text())
    require(p["status"] == "cpu_qualified_before_model" and
            p["protocol"] == bind(PROTOCOL) and p["admission"] == bind(ADMISSION) and
            p["producer"] == bind(Path(__file__)), "preflight binding changed")
    for c in p["source_captures"]:
        require(bind(c["maintained"]["path"]) == c["maintained"] and
                bind(c["capture"]["path"]) == c["capture"],
                "direct source capture changed")
    return a, old_a, p


def gpu_child():
    a, old_a, p = checked()
    require(not (OUT / "launch.json").exists() and not (OUT / "receipt.json").exists(),
            "attempt already launched; no automatic retry")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device("cuda:0")
    handles = []
    counts = {"model_forwards": 0, "vision_forwards": 0, "generated_tokens": 0}
    receipt = {"status": "running", "pid": os.getpid(), "begun_unix": time.time(),
               "admission": bind(ADMISSION), "preflight": bind(PREFLIGHT),
               "producer": bind(Path(__file__)), "counts": counts, "cells": []}
    write_new(OUT / "launch.json", receipt)
    active: dict = {}
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = car.base.load_model("untied", device)
        expected = p["source_identity"]
        require({k: v for k, v in identity.items() if k != "loader_source"} ==
                {k: v for k, v in expected.items() if k != "loader_source"} and
                all(identity["loader_source"][k] == expected["loader_source"][k]
                    for k in ("sha256", "size_bytes")) and
                identity["loader_source"]["path"] ==
                a["loader_crosswalk"]["maintained"]["path"],
                "effective model/maintained loader changed")
        model = q.model.eval()
        batch, raw, trace, sr, _ = car.prior.source(q, old_a, device)
        require(sr["input_identity"] == p["input_identity"] and
                int(q.tokenizer.pad_token_id) == p["pad_id"],
                "GPU original source/tokenizer changed")
        fulls = {o: car.make_full(model, batch, raw, p["pad_id"], o)
                 for o in ("AF", "FF")}
        require({o: {k: car.base.tensor_hash(v) for k, v in full.items()
                     if isinstance(v, torch.Tensor)} for o, full in fulls.items()} ==
                p["full_inputs"], "GPU AF/FF full inputs differ from CPU preflight")
        attentions = [layer.self_attn for layer in model.model.language_model.layers]
        require(len(attentions) == LAYERS and
                all(isinstance(x, modeling_qwen3_vl.Qwen3VLTextAttention)
                    for x in attentions), "actual 28-layer attention route changed")

        def before_model(_module, _args, kwargs):
            counts["model_forwards"] += 1
            require(counts["model_forwards"] <= 22 and
                    active.get("name") == NAMES[counts["model_forwards"] - 1],
                    "unexpected model call/order")
            car.verify_actual_input(kwargs, active["input"], media=active["media"])
            active["top_input"] = {k: car.base.tensor_hash(kwargs[k]) for k in
                                   ("input_ids", "attention_mask", "position_ids",
                                    "cache_position")}

        def before_vision(_module, _args):
            counts["vision_forwards"] += 1
            require(active.get("media") and counts["vision_forwards"] <= 4,
                    "unexpected vision call")

        def on_rotary(_module, args, output):
            require(len(args) >= 2 and
                    torch.equal(args[1], active["input"]["position_ids"]),
                    "actual rotary positions differ from source")
            cos, sin = output
            n = active["input"]["input_ids"].shape[1]
            require(cos.shape == sin.shape == (4, n, DIM) and
                    not active["rotary_values"], "actual rotary shape/count changed")
            active["rotary_values"].append((cos, sin))
            active["rotary_hashes"] = {"cos": car.base.tensor_hash(cos),
                                       "sin": car.base.tensor_hash(sin)}

        handles.extend((model.register_forward_pre_hook(before_model, with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision),
                        model.model.language_model.rotary_emb.register_forward_hook(on_rotary)))
        for i, attention in enumerate(attentions):
            def at_entry(_module, _args, kwargs, *, index=i):
                mask = kwargs.get("attention_mask")
                car.verify_attention(mask, active["expected_mask"])
                embed = kwargs.get("position_embeddings")
                require(len(active["rotary_values"]) == 1 and
                        isinstance(embed, tuple) and len(embed) == 2 and
                        torch.equal(embed[0], active["rotary_values"][0][0]) and
                        torch.equal(embed[1], active["rotary_values"][0][1]),
                        "actual attention rotary consumer changed")
                record = {"layer": index,
                          "mask_sha256": car.base.tensor_hash(mask),
                          "rotary": active["rotary_hashes"]}
                if active["kind"] == "suffix":
                    cache = active["cache"]
                    require(kwargs.get("past_key_values") is cache and
                            cache.get_seq_length(index) == WIDTH and
                            torch.equal(kwargs.get("cache_position"),
                                        active["input"]["cache_position"]),
                            "actual suffix cache slot changed")
                    record["segments"] = old.older_layer_selected(
                        cache.layers[index], active["base_segments"][index],
                        active["donor_blocks"]["layers"][index],
                        base_origin=active["base"], donor_origin=active["donor"],
                        mode=active["mode"])
                active["attn"].append(record)
            handles.append(attention.register_forward_pre_hook(at_entry, with_kwargs=True))
        for i, layer in enumerate(model.model.language_model.layers):
            def at_output(_module, args, *, index=i, current_layer=layer):
                if active["kind"] != "suffix":
                    return
                cache = active["cache"]
                require(cache.get_seq_length(index) == END and
                        {axis: car.base.tensor_hash(getattr(cache.layers[index], axis)[:,:,:WIDTH,:])
                         for axis in ("keys", "values")} == active["patched_digest"][index],
                        "historical cache changed during suffix")
                row = {axis: car.base.tensor_hash(
                    getattr(cache.layers[index], axis)[[0, 1, 3], :, WIDTH:END, :])
                    for axis in ("keys", "values")}
                active["after"].append({"layer": index, "companion_suffix": row,
                                        "historical_digest": active["patched_digest"][index]})
                observe_headout(active, current_layer, index, args[0])
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(at_output))

        captures: dict[str, list[dict[str, torch.Tensor]]] = {
            "AF": [{} for _ in range(LAYERS)], "FF": [{} for _ in range(LAYERS)]}
        full_vectors: dict[str, torch.Tensor] = {}
        caches: dict[str, DynamicCache] = {}
        segments: dict = {}
        blocks: dict = {}
        digests: dict = {}
        vectors: dict[str, torch.Tensor] = {}
        consumers: dict = {}

        def invoke(cell, inputs, *, cache=None):
            name = cell["name"]
            index = NAMES.index(name)
            kind = "suffix" if index >= 4 else "prefill" if index >= 2 else "full"
            base = cell["base_origin"]
            mode = ("anchor" if cell["kind"] == "clamp_native" else
                    cell["kind"].removeprefix("clamp_") if cell["clamp"] else
                    old.cell_mode(name) if kind == "suffix" else kind)
            donor = (cell["older_K_origin"] if mode in ("joint", "K") else
                     cell["older_V_origin"] if mode == "V" else base)
            expected_mask = (car.expected_suffix_mask(fulls[base]) if kind == "suffix"
                             else car.base.native_4d(inputs["attention_mask"]))
            active.clear()
            active.update(name=name, kind=kind, mode=mode, base=base, donor=donor,
                          clamp=cell["clamp"], capture=cell["capture_native_header"],
                          input=inputs, media=bool(cell["vision"]), cache=cache,
                          expected_mask=expected_mask, attn=[], after=[],
                          rotary_values=[], rotary_hashes={},
                          qkv=[{} for _ in range(LAYERS)],
                          qkv_hashes=[{} for _ in range(LAYERS)],
                          headouts=[], headout_packets=[])
            if kind == "suffix":
                active["base_segments"] = segments[base]
                active["donor_blocks"] = blocks[donor]
                with old.scope(cache, digests[base], mode=mode,
                               base_origin=base, donor_origin=donor,
                               donor_blocks=blocks[donor]):
                    active["patched_digest"] = cache_digest(cache)
                    for layer_index in range(LAYERS):
                        old.older_layer_selected(cache.layers[layer_index],
                            segments[base][layer_index], blocks[donor]["layers"][layer_index],
                            base_origin=base, donor_origin=donor, mode=mode)
                    with header_hooks(model, active, captures):
                        with torch.inference_mode():
                            logits = model(**inputs).logits[:, -1, :].detach().float().cpu()
            else:
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, :].detach().float().cpu()
            torch.cuda.synchronize(device)
            require(logits.shape == (4, 152670) and torch.isfinite(logits).all().item() and
                    [x["layer"] for x in active["attn"]] == list(range(LAYERS)) and
                    len(active["rotary_values"]) == 1 and
                    (kind != "suffix" or
                     ([x["layer"] for x in active["after"]] == list(range(LAYERS)) and
                      [x["layer"] for x in active["headout_packets"]] == list(range(LAYERS)) and
                      all(set(row) == {"Q", "K", "V"} for row in active["qkv_hashes"]))),
                    f"{name} actual vector/consumer evidence incomplete")
            directory = OUT / "cells" / f"{index+1:02d}-{name}"
            directory.mkdir(parents=True, exist_ok=False)
            torch.save(logits, directory / "full-batch-vocabulary.pt")
            write_new(directory / "input.json", {k: v.detach().cpu().tolist()
                for k, v in inputs.items() if k in
                ("input_ids", "attention_mask", "position_ids", "cache_position")})
            if kind == "suffix":
                torch.save(active["headouts"], directory / "target-headouts.pt")
            consumer = {"name": name, "kind": kind, "mode": mode,
                        "base_origin": base, "donor_origin": donor,
                        "clamp": cell["clamp"], "capture_native_header": cell["capture_native_header"],
                        "actual_input": active["top_input"],
                        "rotary": active["rotary_hashes"],
                        "attention": active["attn"], "after_suffix": active["after"],
                        "qkv": active["qkv_hashes"] if kind == "suffix" else [],
                        "headout": active["headout_packets"],
                        "patched_digest": active.get("patched_digest"),
                        "restored_digest": cache_digest(cache) if kind == "suffix" else None,
                        "expected_mask_sha256": car.base.tensor_hash(expected_mask)}
            write_new(directory / "consumer.json", consumer)
            record = {"call": index+1, "name": name, "origins": axes(cell),
                      "clamp": cell["clamp"],
                      "input": bind(directory / "input.json"),
                      "vector": bind(directory / "full-batch-vocabulary.pt"),
                      "consumer": bind(directory / "consumer.json"),
                      "headouts": bind(directory / "target-headouts.pt") if kind == "suffix" else None,
                      "counts_after": dict(counts)}
            receipt["cells"].append(record)
            return logits, consumer

        for cell in a["cells"]:
            name, base = cell["name"], cell["base_origin"]
            index = cell["call"] - 1
            require(index == len(receipt["cells"]) and
                    (index < 10 or all(n in vectors for n in NAMES[4:10])) and
                    (index < 14 or all(n in vectors for n in NAMES[:14])) and
                    (index < 16 or all(n in vectors for n in NAMES[:16])),
                    "finite qualification barrier/order changed")
            if index < 2:
                logits, consumer = invoke(cell, fulls[base])
                full_vectors[base] = logits
                source_ref = torch.load(a["saved_full_vector_references"][
                    "native_AF" if base == "AF" else "FF"]["path"],
                    map_location="cpu", weights_only=True)["logits"]
                error = float((logits - source_ref).abs().max())
                require(error <= TOL, f"{name} original full vector differs: {error}")
                indexes = range(4) if base == "AF" else (0, 1, 3)
                parity = [car.base._trace_compare(
                    logits=logits[j], trace=trace, batch_index=j,
                    absolute_offset=22, token_id=raw[j]["token_ids"][22],
                    role="original_AF_x1" if base == "AF" and j == TARGET else
                         "source_companion_x1", atol=TOL) for j in indexes]
                require(all(x["passed"] for x in parity),
                        f"{name} original trace parity failed")
                receipt["cells"][-1].update(original_full_error=error,
                                             source_trace_parity=parity)
            elif index < 4:
                cache = DynamicCache()
                inputs = car.split_inputs(fulls[base], cache, "prefill")
                logits, consumer = invoke(cell, inputs, cache=cache)
                require(cache.get_seq_length() == WIDTH,
                        f"{name} prefill width changed")
                caches[base] = cache
                segments[base] = car.segments(cache)
                blocks[base] = car.blocks(cache, base)
                digests[base] = cache_digest(cache)
                receipt["cells"][-1]["cache_digest"] = digests[base]
                if index == 3:
                    require(all(segments["AF"][i][axis][span] ==
                                segments["FF"][i][axis][span]
                                for i in range(LAYERS) for axis in ("keys", "values")
                                for span in ("prompt", "companions")),
                            "AF/FF prior prehistory or companions changed")
                    torch.save(blocks, OUT / "historical-blocks.pt")
                    write_new(OUT / "prefill-origins.json",
                              {"segments": segments, "digests": digests,
                               "blocks": bind(OUT / "historical-blocks.pt")})
            else:
                inputs = car.split_inputs(fulls[base], caches[base], "suffix")
                logits, consumer = invoke(cell, inputs, cache=caches[base])
                require(consumer["restored_digest"] == digests[base],
                        f"{name} cache restoration changed")
                if cell["capture_native_header"]:
                    require(all(set(row) == {"Q", "K", "V"} for row in captures[base]),
                            f"{name} native Q/K/V capture incomplete")
                    path = OUT / "captures" / f"native-{base}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"base": base,
                                "layers": [{axis: value.detach().cpu() for axis, value in row.items()}
                                           for row in captures[base]]}, path)
                    receipt["cells"][-1]["native_header_capture"] = bind(path)
                anchor = vectors.get("anchor_" + base)
                if anchor is not None:
                    companion_error = max(float((logits[j] - anchor[j]).abs().max())
                                          for j in (0, 1, 3))
                    require(companion_error <= TOL and
                            [x["companion_suffix"] for x in consumer["after_suffix"]] ==
                            [x["companion_suffix"] for x in
                             consumers["anchor_" + base]["after_suffix"]],
                            f"{name} companion logits/current K/V changed")
                    receipt["cells"][-1]["companion_anchor_max_abs"] = companion_error
                if cell["kind"] == "anchor":
                    accepted = torch.load(a["anchor_references"][base]["path"],
                                          map_location="cpu", weights_only=True)
                    error = max(float((logits - full_vectors[base]).abs().max()),
                                float((logits - accepted).abs().max()))
                    require(error <= TOL, f"{name} full/cache anchor mismatch: {error}")
                    receipt["cells"][-1]["full_and_saved_anchor_max_abs"] = error
                if cell["kind"] == "sham":
                    error = float((logits - anchor).abs().max())
                    require(error <= TOL, f"{name} older-write sham mismatch: {error}")
                    receipt["cells"][-1]["older_sham_max_abs"] = error
                if cell["kind"] == "joint":
                    ref = a["joint_references"][name]
                    prior_vec = torch.load(ref["vector"]["path"],
                                           map_location="cpu", weights_only=True)
                    error = float((logits - prior_vec).abs().max())
                    require(error <= TOL and
                            json.loads(Path(receipt["cells"][-1]["input"]["path"]).read_text()) ==
                            json.loads(Path(ref["input"]["path"]).read_text()),
                            f"{name} complementary joint reference mismatch: {error}")
                    receipt["cells"][-1]["complementary_joint_max_abs"] = error
                if cell["kind"] == "clamp_native":
                    error = float((logits - anchor).abs().max())
                    require(error <= TOL, f"{name} all-four clamp identity mismatch: {error}")
                    receipt["cells"][-1]["native_clamp_all_four_max_abs"] = error
            if index < 14:
                reference = a["adaptive_references"][name]
                accepted = torch.load(reference["vector"]["path"],
                                      map_location="cpu", weights_only=True)
                error = float((logits - accepted).abs().max())
                require(error <= TOL and
                        json.loads(Path(receipt["cells"][-1]["input"]["path"]).read_text()) ==
                        json.loads(Path(reference["input"]["path"]).read_text()),
                        f"{name} adaptive all-four reference/input mismatch: {error}")
                receipt["cells"][-1]["accepted_adaptive_max_abs"] = error
            vectors[name] = logits
            consumers[name] = consumer
        require(counts == {"model_forwards": 22, "vision_forwards": 4,
                           "generated_tokens": 0}, "finite 22/4/0 counts changed")
        receipt["effective_identity"] = identity
        receipt["status"] = "candidate_raw_complete"
    except Exception as exc:
        receipt["status"] = "technical_failure"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc(),
                              "active_cell": active.get("name")}
        if active.get("headouts"):
            path = OUT / "partial-headouts.pt"
            torch.save(active["headouts"], path)
            receipt["failure"]["partial_headouts"] = bind(path)
        if active.get("headout_packets"):
            path = OUT / "partial-consumer.json"
            write_new(path, {"name": active.get("name"),
                             "headout": active["headout_packets"],
                             "qkv": active.get("qkv_hashes", []),
                             "attention": active.get("attn", []),
                             "after_suffix": active.get("after", [])})
            receipt["failure"]["partial_consumer"] = bind(path)
    finally:
        for handle in handles:
            handle.remove()
        receipt["internal_seconds"] = time.monotonic() - started
        receipt["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch.cuda.is_available():
            receipt["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            receipt["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        receipt["artifact_bytes_before_receipt"] = sum(
            p.stat().st_size for p in OUT.rglob("*") if p.is_file())
        receipt["terminal_pid"] = os.getpid()
        write_new(OUT / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "counts": counts,
                          "failure": receipt.get("failure", {}).get("message")}))
    require(receipt["status"] == "candidate_raw_complete",
            "technical failure; no automatic retry")


def run_parent():
    checked()
    require(not (OUT / "launch.json").exists() and not (OUT / "outer.json").exists(),
            "attempt already launched; no automatic retry")
    OUT.mkdir(parents=True, exist_ok=True)
    command = ["python", "-B", "-m",
               "probes.training_set_completion.recurrence_older_kv_header_clamp.run", "gpu"]
    began, unix = time.monotonic(), time.time()
    with (OUT / "stdout.log").open("x") as stdout, (OUT / "stderr.log").open("x") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        code = child.wait()
    outer = {"command": command, "started_unix": unix,
             "outer_seconds": time.monotonic() - began,
             "child_pid": child.pid, "returncode": code, "terminal": True,
             "stdout": bind(OUT / "stdout.log"), "stderr": bind(OUT / "stderr.log")}
    write_new(OUT / "outer.json", outer)
    print(json.dumps(outer))
    require(code == 0, "terminal child failure; no automatic retry")


def reduce_vectors(vectors):
    """Frozen full-vocabulary FP64 TVs and descriptive token readout."""
    probabilities = {n: torch.softmax(v[TARGET].double(), dim=-1)
                     for n, v in vectors.items() if not n.startswith("prefill_")}
    def tv(x, y):
        return float((probabilities[x] - probabilities[y]).abs().sum() / 2)
    def predicate(value, threshold, direction):
        if value is None or abs(value - threshold) <= 1e-6:
            return None
        return value < threshold if direction == "below" else value > threshold
    bases = {}
    descriptive = {}
    for base, donor in (("AF", "FF"), ("FF", "AF")):
        native = "anchor_" + base
        joint = f"joint_{base}_from_{donor}"
        clamp = {"N": "clamp_native_" + base,
                 "J": f"clamp_joint_{base}_from_{donor}",
                 "V": f"clamp_V_{base}_from_{donor}",
                 "K": f"clamp_K_{base}_from_{donor}"}
        distance = tv(native, joint)
        clamp_distance = tv(clamp["N"], clamp["J"])
        denom_ok = distance > 1e-6 and abs(distance - 1e-6) > 1e-6
        c_denom_ok = clamp_distance > 1e-6 and abs(clamp_distance - 1e-6) > 1e-6
        endpoint = tv(clamp["J"], joint) / distance if denom_ok else None
        displacement = tv(clamp["J"], clamp["N"]) / distance if denom_ok else None
        ratios = ({
            "r_V": tv(clamp["V"], clamp["J"]) / clamp_distance,
            "r_K": tv(clamp["K"], clamp["J"]) / clamp_distance,
            "n_V": tv(clamp["V"], clamp["N"]) / clamp_distance,
            "n_K": tv(clamp["K"], clamp["N"]) / clamp_distance,
        } if c_denom_ok else {k: None for k in ("r_V", "r_K", "n_V", "n_K")})
        components = ({
            "r_V": predicate(ratios["r_V"], .5, "below" if base == "AF" else "above"),
            "r_K": predicate(ratios["r_K"], .5, "below" if base == "AF" else "above"),
            "n_V": predicate(ratios["n_V"], .2, "below") if base == "FF" else None,
            "n_K": predicate(ratios["n_K"], .2, "below") if base == "FF" else None,
        })
        bases[base] = {"native": native, "joint": joint,
                       "adaptive_D": distance, "clamped_D": clamp_distance,
                       "endpoint_e": endpoint, "displacement_t": displacement,
                       "retention": predicate(endpoint, .5, "below"),
                       "collapse": predicate(displacement, .2, "below"),
                       "component_ratios": ratios,
                       "component_predicates": components,
                       "component_pattern": (all(components[k] is True for k in
                                                (("r_V", "r_K") if base == "AF" else
                                                 ("r_V", "r_K", "n_V", "n_K")))
                                             if c_denom_ok and all(
                                                components[k] is not None for k in
                                                (("r_V", "r_K") if base == "AF" else
                                                 ("r_V", "r_K", "n_V", "n_K")))
                                             else None),
                       "clamped_corners": {label: {
                           "cell": name,
                           "TV_to_native": tv(name, native),
                           "TV_to_joint": tv(name, joint),
                           "TV_to_adaptive_counterpart": tv(name,
                               {"N": native, "J": joint,
                                "V": f"V_{base}_from_{donor}",
                                "K": f"K_{base}_from_{donor}"}[label])}
                           for label, name in clamp.items()}}
        for name in clamp.values():
            z = vectors[name][TARGET].double()
            p = probabilities[name]
            top = torch.topk(z, 2)
            descriptive[name] = {
                "winner": int(top.indices[0]), "runner": int(top.indices[1]),
                "gap": float(top.values[0] - top.values[1]),
                "z_151671_minus_151670": float(z[151671] - z[151670]),
                "fixed_tokens": {str(j): {
                    "probability": float(p[j]), "rank": int((z > z[j]).sum()) + 1}
                    for j in (151670, 151671)}}
    retention = all(bases[b]["retention"] is True for b in ("AF", "FF"))
    collapse = all(bases[b]["collapse"] is True for b in ("AF", "FF"))
    if retention:
        shared = "bidirectional_joint_endpoint_retention"
    elif collapse:
        shared = "bidirectional_displacement_collapse"
    elif any(bases[b]["retention"] is None or bases[b]["collapse"] is None
             for b in ("AF", "FF")):
        shared = "numerical_or_denominator_HOLD"
    else:
        shared = "mixed_or_changed"
    component = (retention and all(bases[b]["component_pattern"] is True
                                   for b in ("AF", "FF")))
    if any(bases[b]["component_pattern"] is None for b in ("AF", "FF")):
        component_status = "unresolved"
    else:
        component_status = "retained_pattern" if component else "not_retained"
    return {"bases": bases, "shared_outcome": shared,
            "shared_retention_pass": retention,
            "shared_collapse_comparator_pass": collapse,
            "secondary_component_pattern": component_status,
            "descriptive_clamped_cells": descriptive}


def readback():
    a, old_a, p = checked()
    require(not (OUT / "readback.json").exists() and
            not (UNIT / "candidate-results.md").exists(),
            "candidate readback already exists")
    receipt = json.loads((OUT / "receipt.json").read_text())
    outer = json.loads((OUT / "outer.json").read_text())
    require(receipt["status"] == "candidate_raw_complete" and
            receipt["admission"] == bind(ADMISSION) and
            receipt["preflight"] == bind(PREFLIGHT) and
            receipt["producer"] == bind(Path(__file__)) and
            receipt["counts"] == {"model_forwards": 22,
                                   "vision_forwards": 4,
                                   "generated_tokens": 0} and
            outer["returncode"] == 0 and outer["terminal"] and
            outer["child_pid"] == receipt["terminal_pid"] and
            not Path(f"/proc/{outer['child_pid']}").exists(),
            "terminal model/source/count binding changed")
    record_guard(receipt["cells"], a["cells"])
    q = car.base.load_qwen_components_from_options(car.base.QwenLoadOptions(
        base_model=str(car.base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "cold reader loaded model")
    batch, raw, trace, sr, _ = car.prior.source(q, old_a, torch.device("cpu"))
    require(sr["input_identity"] == p["input_identity"] and
            int(q.tokenizer.pad_token_id) == p["pad_id"],
            "cold original source changed")
    fulls = {o: car.make_full(car.base.ConfigOnlyRope(), batch, raw, p["pad_id"], o)
             for o in ("AF", "FF")}
    pre = json.loads((OUT / "prefill-origins.json").read_text())
    require(pre["blocks"] == bind(OUT / "historical-blocks.pt") and
            set(pre["segments"]) == set(pre["digests"]) == {"AF", "FF"},
            "cold historical prefill origin changed")
    snapshots = torch.load(pre["blocks"]["path"], map_location="cpu", weights_only=True)
    require(set(snapshots) == {"AF", "FF"}, "cold historical donor set changed")
    for base in ("AF", "FF"):
        require(snapshots[base]["origin"] == base and
                len(snapshots[base]["layers"]) ==
                len(pre["segments"][base]) == len(pre["digests"][base]) == LAYERS,
                "cold historical donor layers changed")
        for i, row in enumerate(snapshots[base]["layers"]):
            for span in ("older", "latest"):
                for axis in ("keys", "values"):
                    tensor = row[span][axis]
                    require(tensor.shape == (KV_HEADS, 9, DIM) and
                            torch.isfinite(tensor).all().item() and
                            car.base.tensor_hash(tensor) ==
                            pre["segments"][base][i][axis][span],
                            "cold historical donor shape/hash changed")
    require(all(pre["segments"]["AF"][i][axis][span] ==
                pre["segments"]["FF"][i][axis][span]
                for i in range(LAYERS) for axis in ("keys", "values")
                for span in ("prompt", "companions")),
            "cold AF/FF prompt or companion history changed")
    captures = {}
    for base in ("AF", "FF"):
        cell = receipt["cells"][NAMES.index("anchor_" + base)]
        path = cell["native_header_capture"]["path"]
        require(bind(path) == cell["native_header_capture"],
                "cold native header capture binding changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(saved["base"] == base and len(saved["layers"]) == LAYERS,
                "cold native header capture base/layers changed")
        for layer in saved["layers"]:
            require(set(layer) == {"Q", "K", "V"} and
                    layer["Q"].shape == (4, Q_HEADS, DIM) and
                    layer["K"].shape == (4, KV_HEADS, DIM) and
                    layer["V"].shape == (4, KV_HEADS * DIM) and
                    all(x.dtype == torch.float32 and torch.isfinite(x).all().item()
                        for x in layer.values()),
                    "cold native header Q/K/V shape/nonfinite changed")
        captures[base] = saved["layers"]
    vectors, consumers, records = {}, {}, []
    for i, cell in enumerate(receipt["cells"]):
        admitted = a["cells"][i]
        name, base = cell["name"], admitted["base_origin"]
        for key in ("input", "vector", "consumer"):
            require(bind(cell[key]["path"]) == cell[key],
                    f"cold {name} {key} binding changed")
        if i >= 4:
            require(bind(cell["headouts"]["path"]) == cell["headouts"],
                    f"cold {name} headout binding changed")
        else:
            require(cell["headouts"] is None, "cold full/prefill headouts invented")
        saved_input = json.loads(Path(cell["input"]["path"]).read_text())
        consumer = json.loads(Path(cell["consumer"]["path"]).read_text())
        vector = torch.load(cell["vector"]["path"], map_location="cpu", weights_only=True)
        require(vector.shape == (4, 152670) and torch.isfinite(vector).all().item() and
                consumer["name"] == name and consumer["base_origin"] == base and
                consumer["clamp"] is admitted["clamp"] and
                [x["layer"] for x in consumer["attention"]] == list(range(LAYERS)) and
                len(consumer["after_suffix"]) == (LAYERS if i >= 4 else 0),
                f"cold {name} vector/consumer schema changed")
        expected = fulls[base] if i < 2 else car.split_inputs(
            fulls[base], object(), "prefill" if i < 4 else "suffix")
        for key in ("input_ids", "attention_mask", "position_ids", "cache_position"):
            require(saved_input[key] == expected[key].cpu().tolist() and
                    consumer["actual_input"][key] == car.base.tensor_hash(expected[key]),
                    f"cold {name} source/actual {key} changed")
        mask = (car.expected_suffix_mask(fulls[base]) if i >= 4 else
                car.base.native_4d(expected["attention_mask"]))
        mask_hash = car.base.tensor_hash(mask)
        require(consumer["expected_mask_sha256"] == mask_hash and
                all(row["mask_sha256"] == mask_hash for row in consumer["attention"]),
                f"cold {name} actual 28-layer mask changed")
        if i >= 4:
            mode = ("anchor" if admitted["kind"] == "clamp_native" else
                    admitted["kind"].removeprefix("clamp_") if admitted["clamp"] else
                    old.cell_mode(name))
            donor = (admitted["older_K_origin"] if mode in ("joint", "K") else
                     admitted["older_V_origin"] if mode == "V" else base)
            require(consumer["mode"] == mode and consumer["donor_origin"] == donor and
                    consumer["restored_digest"] == pre["digests"][base] and
                    len(consumer["patched_digest"]) == LAYERS and
                    len(consumer["qkv"]) == len(consumer["headout"]) == LAYERS,
                    f"cold {name} mode/donor/restoration changed")
            headouts = torch.load(cell["headouts"]["path"],
                                  map_location="cpu", weights_only=True)
            require(len(headouts) == LAYERS, f"cold {name} saved headout layers changed")
            for layer, record in enumerate(consumer["attention"]):
                for axis, origin_key in (("keys", "older_K_origin"),
                                         ("values", "older_V_origin")):
                    got = record["segments"][axis]
                    base_row = pre["segments"][base][layer][axis]
                    wanted = car.base.tensor_hash(
                        snapshots[admitted[origin_key]]["layers"][layer]["older"][axis])
                    require(all(got[span] == base_row[span] for span in
                                ("prompt", "latest", "companions")) and
                            got["older"] == wanted and
                            consumer["patched_digest"][layer][axis] ==
                                consumer["after_suffix"][layer]["historical_digest"][axis],
                            f"cold {name} layer {layer} older {axis}/complement changed")
                qkv = consumer["qkv"][layer]
                require(set(qkv) == {"Q", "K", "V"},
                        f"cold {name} layer {layer} Q/K/V incomplete")
                if admitted["clamp"] or admitted["capture_native_header"]:
                    require(all(qkv[axis]["target"] ==
                                car.base.tensor_hash(captures[base][layer][axis])
                                for axis in ("Q", "K", "V")),
                            f"cold {name} layer {layer} own-base capture not consumed")
                packet = consumer["headout"][layer]
                actual, reconstructed = headouts[layer]["actual"], headouts[layer]["reconstructed"]
                error = float((actual.double() - reconstructed).abs().max())
                scale = max(1.0, float(actual.abs().max()),
                            float(reconstructed.abs().max()))
                require(packet["layer"] == layer and actual.shape == reconstructed.shape ==
                        (Q_HEADS, 4, DIM) and
                        packet["actual_headout_hash"] == car.base.tensor_hash(actual) and
                        packet["reconstructed_headout_hash"] ==
                        car.base.tensor_hash(reconstructed) and
                        abs(packet["max_abs_error"] - error) <= 1e-12 and
                        abs(packet["scale"] - scale) <= 1e-12 and
                        abs(packet["bound"] - HEAD_REL * scale) <= 1e-12 and
                        error <= HEAD_REL * scale and
                        packet["postK_max_abs_error"] <= POSTK_TOL and
                        packet["V_max_abs_error"] == 0,
                        f"cold {name} layer {layer} saved headout gate changed")
        else:
            require(consumer["patched_digest"] is None and
                    consumer["restored_digest"] is None and
                    consumer["qkv"] == consumer["headout"] == [],
                    "cold full/prefill suffix evidence invented")
        vectors[name], consumers[name] = vector, consumer
        records.append({"call": i+1, "name": name, "origins": axes(admitted),
                        "clamp": admitted["clamp"], "input": cell["input"],
                        "vector": cell["vector"], "consumer": cell["consumer"],
                        "headouts": cell["headouts"]})
    for name in NAMES[:14]:
        accepted = a["adaptive_references"][name]
        vector = torch.load(accepted["vector"]["path"], map_location="cpu",
                            weights_only=True)
        require(float((vectors[name] - vector).abs().max()) <= TOL and
                json.loads(Path(accepted["input"]["path"]).read_text()) ==
                json.loads(Path(records[NAMES.index(name)]["input"]["path"]).read_text()),
                f"cold adaptive {name} accepted all-four vector/input changed")
    for name, base in (("full_AF", "AF"), ("full_FF", "FF")):
        accepted = torch.load(a["saved_full_vector_references"][
            "native_AF" if base == "AF" else "FF"]["path"],
            map_location="cpu", weights_only=True)["logits"]
        require(float((vectors[name] - accepted).abs().max()) <= TOL,
                "cold original full-prefix reference changed")
        indexes = range(4) if base == "AF" else (0, 1, 3)
        require(all(car.base._trace_compare(
            logits=vectors[name][j], trace=trace, batch_index=j,
            absolute_offset=22, token_id=raw[j]["token_ids"][22],
            role="cold_AF_or_companion", atol=TOL)["passed"] for j in indexes),
            "cold original source trace changed")
    for base in ("AF", "FF"):
        anchor = vectors["anchor_" + base]
        accepted = torch.load(a["anchor_references"][base]["path"],
                              map_location="cpu", weights_only=True)
        require(max(float((anchor - vectors["full_" + base]).abs().max()),
                    float((anchor - accepted).abs().max()),
                    float((vectors["sham_" + base] - anchor).abs().max()),
                    float((vectors["clamp_native_" + base] - anchor).abs().max())) <= TOL,
                "cold full/cache/sham/native-clamp all-four gate changed")
    for name in NAMES[8:]:
        base = a["cells"][NAMES.index(name)]["base_origin"]
        require(max(float((vectors[name][j] - vectors["anchor_" + base][j]).abs().max())
                    for j in (0, 1, 3)) <= TOL and
                [x["companion_suffix"] for x in consumers[name]["after_suffix"]] ==
                [x["companion_suffix"] for x in
                 consumers["anchor_" + base]["after_suffix"]] and
                all(consumers[name]["qkv"][layer][axis]["companions"] ==
                    consumers["anchor_" + base]["qkv"][layer][axis]["companions"]
                    for layer in range(LAYERS) for axis in ("Q", "K", "V")),
                f"cold {name} companion suffix Q/K/V/logits changed")
    for name in ("joint_AF_from_FF", "joint_FF_from_AF"):
        reference = a["joint_references"][name]
        accepted = torch.load(reference["vector"]["path"],
                              map_location="cpu", weights_only=True)
        require(float((vectors[name] - accepted).abs().max()) <= TOL,
                "cold complementary joint accepted reference changed")
    reduced = reduce_vectors(vectors)
    result = {"status": "candidate_cold_readback_passed", "protocol": bind(PROTOCOL),
              "admission": bind(ADMISSION), "preflight": bind(PREFLIGHT),
              "receipt": bind(OUT / "receipt.json"), "outer": bind(OUT / "outer.json"),
              "prefill_origins": bind(OUT / "prefill-origins.json"),
              "captures": {base: receipt["cells"][NAMES.index("anchor_" + base)]
                           ["native_header_capture"] for base in ("AF", "FF")},
              "cells": records, "counts": receipt["counts"], "outcomes": reduced,
              "model_loads": 0, "model_forwards": 0,
              "vision_forwards": 0, "cuda_calls": 0,
              "gpu_seconds_added_by_readback": 0,
              "limit": "Saved-headout cold arithmetic is not a second full-SDPA replay."}
    write_new(OUT / "readback.json", result)
    lines = ["# Older K/V header clamp candidate", "",
             "Status: **candidate cold readback passed; lead acceptance pending.**", "",
             f"Protocol SHA `{SHAS[PROTOCOL]}`; admission SHA `{SHAS[ADMISSION]}`; "
             f"producer SHA `{bind(Path(__file__))['sha256']}`.",
             f"Raw receipt SHA `{result['receipt']['sha256']}`; cold readback SHA "
             f"`{bind(OUT/'readback.json')['sha256']}`. Original refined-03 four requests, "
             "target2 train351017; AF/FF common header replayed, zero generated tokens.", "",
             "## Frozen FP64 full-vocabulary decision", "",
             f"Shared outcome **{reduced['shared_outcome']}**. Bidirectional endpoint "
             f"retention **{reduced['shared_retention_pass']}**; displacement-collapse "
             f"comparator **{reduced['shared_collapse_comparator_pass']}**; secondary "
             f"component pattern **{reduced['secondary_component_pattern']}**.", "",
             "| Base | Adaptive D | Clamped D | e to adaptive joint | t from clamped native | Retention | Collapse | Component |",
             "|---|---:|---:|---:|---:|---|---|---|"]
    for base in ("AF", "FF"):
        row = reduced["bases"][base]
        fmt = lambda x: "undefined" if x is None else f"{x:.12g}"
        lines.append(f"| {base} | {row['adaptive_D']:.12g} | {row['clamped_D']:.12g} | "
                     f"{fmt(row['endpoint_e'])} | {fmt(row['displacement_t'])} | "
                     f"{row['retention']} | {row['collapse']} | {row['component_pattern']} |")
    lines += ["", "Full clamped-corner TVs to same-base native, joint and adaptive counterpart; "
              "all secondary component ratios, full-vocabulary winners/runners/gaps, and fixed "
              "token probabilities/ranks are in the raw cold readback. Neither a mixed result nor "
              "scientific nonpass skipped a fixed component.", "",
              "## Technical qualification and cost", "",
              "Calls1–14 reproduced accepted all-four vectors; native AF/FF header Q/K/V "
              "was freshly captured at all28 layers and four positions. Both separately "
              "executed native clamps matched their anchors before six clamped treatments. "
              "All22 raw vectors/inputs and selected older K/V donors, latest/prehistory/companion "
              "complements, masks, rotary identities, actual Q/K/V outputs, appended K/V, "
              "scale-aware FP64 pre-o_proj reconstruction and finally restoration were checked. "
              "Cold readback recomputed saved-headout error and all scientific TVs in a separate "
              "CPU process; it did not perform a second SDPA forward.", "",
              f"Actual {receipt['counts']['model_forwards']} model / "
              f"{receipt['counts']['vision_forwards']} vision / "
              f"{receipt['counts']['generated_tokens']} generated. Parent outer "
              f"{outer['outer_seconds']:.9f} s; internal {receipt['internal_seconds']:.9f} s. "
              f"Prior sequence {a['prior_sequence_gpu_hours']:.12f} GPUh; cumulative "
              f"{a['prior_sequence_gpu_hours']+outer['outer_seconds']/3600:.12f} GPUh.",
              f"Peak RSS {receipt['rss_peak_kib']} KiB; GPU allocated/reserved "
              f"{receipt['gpu_peak_allocated_bytes']}/{receipt['gpu_peak_reserved_bytes']} B; "
              f"artifact bytes before receipt {receipt['artifact_bytes_before_receipt']}; "
              f"terminal child PID {outer['child_pid']}, exit {outer['returncode']}.", "",
              f"Raw attempt: `{OUT}`. Technical gate tolerances are from the frozen unit; "
              "no scientific threshold was changed. F/F2 physical identity remains HOLD. "
              "The experiment does not identify a natural mediation fraction, literal "
              "address-content circuit or free-row outcome. No self-acceptance or successor.", ""]
    (UNIT / "candidate-results.md").write_text("\n".join(lines))
    print(json.dumps({"status": result["status"],
                      "shared": reduced["shared_outcome"],
                      "outer_seconds": outer["outer_seconds"]}))


def selfcheck():
    a, old_a = contract()
    q = car.base.load_qwen_components_from_options(car.base.QwenLoadOptions(
        base_model=str(car.base.BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    require(q.model is None, "selfcheck loaded model")
    batch, raw, _, _, _ = car.prior.source(q, old_a, torch.device("cpu"))
    checks, _ = cpu_checks(a, old_a, q, batch, raw,
                           int(q.tokenizer.pad_token_id))
    print(json.dumps({"status": "cpu_selfcheck_passed", "checks": checks}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("selfcheck", "preflight", "run", "gpu", "readback"))
    {"selfcheck": selfcheck, "preflight": preflight, "run": run_parent,
     "gpu": gpu_child, "readback": readback}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
