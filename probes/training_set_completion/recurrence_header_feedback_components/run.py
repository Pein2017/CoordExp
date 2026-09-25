"""One admitted 22-call AF/FF Q-only versus K/V-only contrast."""
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
from probes.training_set_completion.recurrence_older_kv_header_clamp import run as whole
from probes.training_set_completion.recurrence_key_phase import rotate
from src.artifacts.source_provenance import preserve_source


ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "research/experiments/2026-09-24-recurrence-header-feedback-components"
PROTOCOL, ADMISSION = UNIT / "unit.md", UNIT / "lead-admission-v1.json"
REPAIR = UNIT / "lead-repair-attempt-002.json"
PREFLIGHT = UNIT / "supporting/attempt-002-preflight.json"
OUT1 = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-header-feedback-components/attempt-001")
OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-header-feedback-components/attempt-002")
SHAS = {PROTOCOL: "1f291c4907e9c0139bbed3dc08d2b55193d5edeea4829e17f9736ee01b15e8a3",
        ADMISSION: "5c98685e3c5e945f81bd28c0b2a7a347d9741f8c1f58c0cd000ad3a299e62300",
        REPAIR: "623e5590992e6941f1db33a0866a716ac2afb0ec6bb08a787dfed97d5758d2e8"}
NAMES = old.NAMES[:10] + (
    "all_native_AF", "all_native_FF", "all_joint_AF_from_FF", "all_joint_FF_from_AF",
    "Q_native_AF", "Q_native_FF", "KV_native_AF", "KV_native_FF",
    "Q_joint_AF_from_FF", "Q_joint_FF_from_AF",
    "KV_joint_AF_from_FF", "KV_joint_FF_from_AF",
)
TARGET, WIDTH, END, LAYERS, KV_HEADS, Q_HEADS, DIM = 2, 1380, 1384, 28, 8, 16, 128
TOL, POSTK_TOL, HEAD_REL = 2e-4, 2e-5, 5e-5
car = old.car
bind, require, write_new = old.bind, old.require, old.write_new
cache_digest = old.cache_digest


def contract():
    for path, sha in SHAS.items():
        require(bind(path)["sha256"] == sha, "frozen header-component contract changed")
    a = json.loads(ADMISSION.read_text())
    repair = json.loads(REPAIR.read_text())
    require(repair["status"] == "lead-admitted-narrow-reference-dispatch-repair-and-one-fresh-attempt" and
            repair["worker_thread"] == a["worker_thread"] and
            repair["worker_model"] == a["worker_model"] and
            repair["worker_effort"] == a["worker_effort"] and
            repair["fresh_attempt"]["output_root"] == str(OUT) and
            repair["fresh_attempt"]["preflight_path"] == str(PREFLIGHT) and
            repair["fresh_attempt"]["cell_names"] == list(NAMES) and
            repair["fresh_attempt"]["max_model_forwards"] == 22 and
            repair["fresh_attempt"]["max_vision_forwards"] == 4 and
            repair["fresh_attempt"]["max_generated_tokens"] == 0 and
            repair["accounting"]["failed_attempt_model_forwards"] == 13 and
            repair["accounting"]["failed_attempt_vision_forwards"] == 4 and
            repair["accounting"]["failed_attempt_outer_seconds"] == 109.01016633212566,
            "narrow repair/attempt002 authority changed")
    for key in ("protocol", "original_admission", "failure_packet", "failed_preflight",
                "failed_receipt", "failed_outer", "failed_producer_capture"):
        require(bind(repair[key]["path"]) == repair[key], f"repair {key} binding changed")
    failed = json.loads(Path(repair["failed_receipt"]["path"]).read_text())
    require(failed["status"] == "technical_failure" and
            failed["counts"] == {"model_forwards": 13, "vision_forwards": 4,
                                  "generated_tokens": 0} and
            failed["terminal_pid"] == json.loads(Path(repair["failed_outer"]["path"]).read_text())["child_pid"] and
            not Path(f"/proc/{failed['terminal_pid']}").exists(),
            "failed attempt charge/terminal binding changed")
    require(a["status"] == "lead-admitted-twenty-two-calls-current-header-axis-split" and
            a["worker_thread"] == "01a0ce4b-9b55-7392-8a25-6a76f9e12c3a" and
            a["worker_model"] == "gpt-6-sol" and a["worker_effort"] == "xhigh" and
            [c["name"] for c in a["cells"]] == list(NAMES) and
            [c["vision"] for c in a["cells"]] == [1] * 4 + [0] * 18 and
            [c["current_mode"] for c in a["cells"]] ==
                ["none"] * 10 + ["all"] * 4 + ["Q_only"] * 2 +
                ["KV_only"] * 2 + ["Q_only"] * 2 + ["KV_only"] * 2 and
            [c["clamped_axes"] for c in a["cells"]] ==
                [[]] * 10 + [["Q", "K", "V"]] * 4 + [["Q"]] * 2 +
                [["K", "V"]] * 2 + [["Q"]] * 2 + [["K", "V"]] * 2 and
            a["max_model_forwards"] == 22 and a["max_vision_forwards"] == 4 and
            a["max_generated_tokens"] == 0 and a["max_model_jobs"] == 1 and
            a["output_root"] == str(OUT1) and a["shape"]["target_index"] == TARGET and
            a["shape"]["older_physical"] == list(old.OLDER) and
            a["shape"]["latest_physical"] == list(old.LATEST) and
            a["shape"]["header_physical"] == [WIDTH, END] and
            a["qualification"]["references_all_four_max_abs"] == TOL and
            a["qualification"]["appended_current_K_max_abs"] == POSTK_TOL and
            a["qualification"]["headout_relative_bound"] == HEAD_REL,
            "finite header-component admission changed")
    for key in ("protocol", "predecessor_acceptance", "predecessor_admission",
                "predecessor_verification", "predecessor_producer",
                "predecessor_preflight", "predecessor_receipt",
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
                  "anchor_references", "native_capture_references"):
        for v in a[group].values():
            require(bind(v["path"]) == v, f"{group} changed")
    for group in ("qualification_references", "joint_references"):
        for cell in a[group].values():
            for v in cell.values():
                if isinstance(v, dict) and "path" in v:
                    require(bind(v["path"]) == v, f"{group} changed")
    old_a = old.contract()
    require(bind(whole.ADMISSION) == a["predecessor_admission"] and
            bind(Path(whole.__file__)) == a["predecessor_producer"] and
            a["source"] == old_a["source"],
            "accepted whole-clamp/source binding changed")
    return a, old_a


def axes(cell):
    return [cell[k] for k in ("base_origin", "older_K_origin", "older_V_origin", "latest_KV_origin")]


def record_guard(cells, admitted):
    require(isinstance(cells, list) and len(cells) == 22, "serialized cell container/count changed")
    for i, (cell, source) in enumerate(zip(cells, admitted, strict=True)):
        require(isinstance(cell, dict) and cell.get("call") == i + 1 and
                cell.get("name") == NAMES[i] and cell.get("origins") == axes(source) and
                cell.get("kind") == source["kind"] and
                cell.get("reference_cell") == source["reference_cell"] and
                cell.get("clamp") is source["clamp"] and
                cell.get("current_mode") == source["current_mode"] and
                cell.get("clamped_axes") == source["clamped_axes"] and
                isinstance(cell.get("input"), dict) and isinstance(cell.get("consumer"), dict),
                "serialized cell order/origins/container changed")


ADAPTIVE_JOINTS = ("joint_AF_from_FF", "joint_FF_from_AF")


def post_forward_references(cell, admitted, a, vector, saved_input):
    """One production/cold dispatch for frozen target references, if any."""
    index = admitted["call"] - 1
    require(0 <= index < 22 and cell["call"] == admitted["call"] and
            cell["name"] == admitted["name"] == NAMES[index] and
            cell["reference_cell"] == admitted["reference_cell"],
            "post-forward cell order or reference alias changed")
    result = {}
    reference_name = cell["reference_cell"]
    if index < 18:
        require(isinstance(reference_name, str) and
                reference_name in a["qualification_references"],
                "required first18 qualification reference missing")
        ref = a["qualification_references"][reference_name]
        accepted = torch.load(ref["vector"]["path"], map_location="cpu", weights_only=True)
        error = float((vector - accepted).abs().max())
        require(error <= TOL and saved_input ==
                json.loads(Path(ref["input"]["path"]).read_text()),
                f"{cell['name']} accepted all-four reference/input mismatch: {error}")
        result["accepted_reference_max_abs"] = error
    else:
        require(reference_name is None, "new treatment has invented target reference")
    if cell["name"] in ADAPTIVE_JOINTS:
        require(index in (8, 9) and cell["kind"] == "joint" and
                cell["name"] in a["joint_references"],
                "required complementary adaptive joint reference missing")
        ref = a["joint_references"][cell["name"]]
        accepted = torch.load(ref["vector"]["path"], map_location="cpu", weights_only=True)
        error = float((vector - accepted).abs().max())
        require(error <= TOL and saved_input ==
                json.loads(Path(ref["input"]["path"]).read_text()),
                f"{cell['name']} complementary adaptive joint mismatch: {error}")
        result["complementary_joint_max_abs"] = error
    return result


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


def selected_axes(mode):
    require(mode in ("none", "all", "Q_only", "KV_only"),
            "wrong current-header axis mode")
    return {"none": (), "all": ("Q", "K", "V"),
            "Q_only": ("Q",), "KV_only": ("K", "V")}[mode]


def verify_axis_output(before, after, expected, *, selected, axis):
    require((torch.equal(after, expected) if selected else torch.equal(after, before)),
            f"{axis} selected capture or free computed tensor changed")


def axis_record_guard(evidence, *, selected, capture, native_hash):
    require(isinstance(evidence, dict) and set(evidence) == {
                "before_target", "after_target", "before_companions",
                "after_companions", "selected"} and
            evidence["selected"] is selected and
            evidence["before_companions"] == evidence["after_companions"] and
            evidence["after_target"] == (
                native_hash if selected or capture else evidence["before_target"]),
            "serialized selected/free current axis changed")


@contextmanager
def header_hooks(model, active, captures):
    """Observe pre/post Q/K/V; replace only the explicit selected axes."""
    handles = []
    try:
        for i, layer in enumerate(model.model.language_model.layers):
            for axis, module in (("Q", layer.self_attn.q_norm),
                                 ("K", layer.self_attn.k_norm),
                                 ("V", layer.self_attn.v_proj)):
                def actuate(_module, _args, output, *, index=i, name=axis):
                    if active["kind"] != "suffix":
                        return output
                    shapes = {"Q": (4, 4, Q_HEADS, DIM),
                              "K": (4, 4, KV_HEADS, DIM),
                              "V": (4, 4, KV_HEADS * DIM)}
                    require(output.shape == shapes[name] and
                            name not in active["qkv_before"][index] and
                            tuple(active["clamped_axes"]) == selected_axes(active["current_mode"]),
                            "wrong pre-actuation axis shape/mode/repeat")
                    before_target = output[TARGET].detach().clone()
                    before_companions = car.base.tensor_hash(output[[0, 1, 3]])
                    active["qkv_before"][index][name] = before_target
                    active["qkv_hashes"][index][name] = {
                        "before_target": car.base.tensor_hash(before_target),
                        "before_companions": before_companions,
                        "selected": name in active["clamped_axes"]}
                    if name in active["clamped_axes"]:
                        require(active["base"] in captures and
                                len(captures[active["base"]]) == LAYERS and
                                name in captures[active["base"]][index],
                                "missing selected own-base native Q/K/V capture")
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
                    selected = name in active["clamped_axes"]
                    expected = (captures[active["base"]][index][name] if selected
                                else active["qkv_before"][index][name])
                    verify_axis_output(active["qkv_before"][index][name], target, expected,
                                       selected=selected, axis=name)
                    require(car.base.tensor_hash(companions) ==
                            active["qkv_hashes"][index][name]["before_companions"],
                            "current Q/K/V hook changed companion")
                    active["qkv"][index][name] = target
                    active["qkv_hashes"][index][name].update(
                        after_target=car.base.tensor_hash(target),
                        after_companions=car.base.tensor_hash(companions))
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
    fixture = {"kind": "suffix", "capture": True, "base": "AF",
               "current_mode": "none", "clamped_axes": [],
               "qkv": [{} for _ in range(LAYERS)],
               "qkv_before": [{} for _ in range(LAYERS)],
               "qkv_hashes": [{} for _ in range(LAYERS)]}
    captures = {"AF": [{} for _ in range(LAYERS)], "FF": [{} for _ in range(LAYERS)]}
    original = {"Q": torch.ones(4, 4, Q_HEADS, DIM),
                "K": torch.ones(4, 4, KV_HEADS, DIM) * 2,
                "V": torch.ones(4, 4, KV_HEADS * DIM) * 3}
    for base, offset in (("AF", 0), ("FF", 10)):
        fixture.update(base=base, capture=True, current_mode="none", clamped_axes=[],
                       qkv=[{} for _ in range(LAYERS)],
                       qkv_before=[{} for _ in range(LAYERS)],
                       qkv_hashes=[{} for _ in range(LAYERS)])
        with header_hooks(model, fixture, captures):
            for layer in model.model.language_model.layers:
                for axis, module in (("Q", layer.self_attn.q_norm),
                                     ("K", layer.self_attn.k_norm),
                                     ("V", layer.self_attn.v_proj)):
                    module(original[axis] + offset)
        require(all(set(row) == {"Q", "K", "V"} for row in captures[base]),
                "CPU native capture did not cover all layers/axes")
    require(not torch.equal(captures["AF"][0]["Q"], captures["FF"][0]["Q"]),
            "CPU bases not distinguishable")
    for base in ("AF", "FF"):
        for mode in ("all", "Q_only", "KV_only"):
            fixture.update(base=base, capture=False, current_mode=mode,
                           clamped_axes=list(selected_axes(mode)),
                           qkv=[{} for _ in range(LAYERS)],
                           qkv_before=[{} for _ in range(LAYERS)],
                           qkv_hashes=[{} for _ in range(LAYERS)])
            with header_hooks(model, fixture, captures):
                for layer in model.model.language_model.layers:
                    for axis, module in (("Q", layer.self_attn.q_norm),
                                         ("K", layer.self_attn.k_norm),
                                         ("V", layer.self_attn.v_proj)):
                        computed = original[axis] + 7
                        consumed = module(computed)
                        wanted = (captures[base][0][axis] if axis in selected_axes(mode)
                                  else computed[TARGET])
                        require(torch.equal(consumed[TARGET], wanted) and
                                torch.equal(consumed[[0, 1, 3]], computed[[0, 1, 3]]),
                                "CPU actual partial caller changed selected/free axis")
            require(all(set(row) == {"Q", "K", "V"} for row in fixture["qkv"]) and
                    all(set(row) == {"Q", "K", "V"} for row in fixture["qkv_hashes"]),
                    "CPU 28-layer axis observer incomplete")
            serialized = json.loads(json.dumps(fixture["qkv_hashes"]))
            for layer in range(LAYERS):
                for axis in ("Q", "K", "V"):
                    axis_record_guard(serialized[layer][axis],
                        selected=axis in selected_axes(mode), capture=False,
                        native_hash=car.base.tensor_hash(captures[base][layer][axis]))
            checks.append(f"actual_28_layer_{base}_{mode}_QKV_pre_post_consumer")
            if mode in ("Q_only", "KV_only"):
                selected = selected_axes(mode)[0]
                free = next(axis for axis in ("Q", "K", "V")
                            if axis not in selected_axes(mode))
                for label, axis, field, value in (
                        ("missing_selected", selected, "after_target", None),
                        ("free_axis_edit", free, "after_target", "bad"),
                        ("companion_edit", free, "after_companions", "bad"),
                        ("wrong_selected_flag", selected, "selected", False)):
                    bad = dict(serialized[0][axis])
                    if value is None:
                        del bad[field]
                    else:
                        bad[field] = value
                    try:
                        axis_record_guard(bad, selected=axis in selected_axes(mode),
                            capture=False,
                            native_hash=car.base.tensor_hash(captures[base][0][axis]))
                    except ValueError:
                        checks.append("serialized_reject_" + label)
                    else:
                        raise AssertionError("serialized axis reader accepted " + label)
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
    fixture.update(base="AF", current_mode="Q_only", clamped_axes=["K", "V"],
                   qkv=[{} for _ in range(LAYERS)],
                   qkv_before=[{} for _ in range(LAYERS)],
                   qkv_hashes=[{} for _ in range(LAYERS)])
    try:
        with header_hooks(model, fixture, captures):
            model.model.language_model.layers[0].self_attn.q_norm(original["Q"])
    except ValueError:
        checks.append("reject_wrong_axis_mask")
    else:
        raise AssertionError("CPU caller admitted wrong axis mask")
    for label, before, after, selected in (
            ("selected_miss", original["Q"][TARGET], original["Q"][TARGET] + 1, True),
            ("free_axis_edit", original["Q"][TARGET], original["Q"][TARGET] + 1, False)):
        try:
            verify_axis_output(before, after, before, selected=selected, axis="Q")
        except ValueError:
            checks.append("reject_" + label)
        else:
            raise AssertionError("axis observer accepted " + label)
    fixture.update(current_mode="Q_only", clamped_axes=["Q"], capture=False,
                   qkv=[{} for _ in range(LAYERS)],
                   qkv_before=[{} for _ in range(LAYERS)],
                   qkv_hashes=[{} for _ in range(LAYERS)])
    removed = captures["AF"][0].pop("Q")
    try:
        with header_hooks(model, fixture, captures):
            model.model.language_model.layers[0].self_attn.q_norm(original["Q"])
    except ValueError:
        checks.append("reject_missing_selected_capture")
    else:
        raise AssertionError("CPU caller admitted missing selected axis")
    finally:
        captures["AF"][0]["Q"] = removed
    fixture["base"] = "absent"
    try:
        with header_hooks(model, fixture, captures):
            model.model.language_model.layers[0].self_attn.q_norm(original["Q"])
    except ValueError:
        checks.append("reject_wrong_base_capture")
    else:
        raise AssertionError("CPU caller admitted wrong base capture")
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
             "origins": axes(c), "kind": c["kind"],
             "reference_cell": c["reference_cell"], "clamp": c["clamp"],
             "current_mode": c["current_mode"], "clamped_axes": c["clamped_axes"]}
            for i, c in enumerate(a["cells"])]
    record_guard(json.loads(json.dumps(good)), a["cells"])
    checks.append("serialized_22_cell_guard_green")
    for label, bad in (("swapped", good[:15] + [good[16], good[15]] + good[17:]),
                       ("dropped", good[:-1]), ("extra", good + [good[-1]]),
                       ("wrong_clamp", good[:14] + [{**good[14], "clamp": False}] + good[15:]),
                       ("wrong_mode", good[:14] + [{**good[14], "current_mode": "all"}] + good[15:]),
                       ("wrong_axes", good[:14] + [{**good[14], "clamped_axes": ["K", "V"]}] + good[15:]),
                       ("wrong_donor", good[:18] + [{**good[18], "origins": ["AF"] * 4}] + good[19:]),
                       ("wrong_container", {"cells": good})):
        try:
            record_guard(json.loads(json.dumps(bad)), a["cells"])
        except ValueError:
            checks.append("serialized_reject_" + label)
        else:
            raise AssertionError("serialized guard accepted " + label)
    for cell in a["cells"]:
        ref_name = cell["reference_cell"] or ("anchor_" + cell["base_origin"])
        ref = a["qualification_references"][ref_name]
        vector = torch.load(ref["vector"]["path"], map_location="cpu", weights_only=True)
        saved_input = json.loads(Path(ref["input"]["path"]).read_text())
        outcome = post_forward_references(cell, cell, a, vector, saved_input)
        require(("accepted_reference_max_abs" in outcome) == (cell["call"] <= 18) and
                ("complementary_joint_max_abs" in outcome) ==
                (cell["name"] in ADAPTIVE_JOINTS),
                "CPU actual post-forward reference dispatch changed")
    checks.append("actual_post_forward_22_cell_reference_dispatch_green")
    source = a["cells"][8]
    ref = a["qualification_references"][source["reference_cell"]]
    vector = torch.load(ref["vector"]["path"], map_location="cpu", weights_only=True)
    saved_input = json.loads(Path(ref["input"]["path"]).read_text())
    bad_vector = vector.clone()
    bad_vector[0, 0] += 1
    missing = dict(a)
    missing["qualification_references"] = dict(a["qualification_references"])
    del missing["qualification_references"][source["reference_cell"]]
    wrong_joint = dict(a)
    wrong_joint["joint_references"] = dict(a["joint_references"])
    wrong_joint["joint_references"][source["name"]] = a["joint_references"][ADAPTIVE_JOINTS[1]]
    for label, cell, admitted, refs, logits, inputs in (
            ("wrong_reference_alias", {**source, "reference_cell": "anchor_AF"},
             source, a, vector, saved_input),
            ("missing_required_reference", source, source, missing, vector, saved_input),
            ("wrong_joint_reference", source, source, wrong_joint, vector, saved_input),
            ("changed_input", source, source, a, vector, {**saved_input, "input_ids": []}),
            ("changed_vector", source, source, a, bad_vector, saved_input),
            ("changed_order", {**source, "call": 10}, source, a, vector, saved_input),
            ("invented_treatment_reference", {**a["cells"][18],
             "reference_cell": "anchor_AF"}, a["cells"][18], a, vector, saved_input)):
        try:
            post_forward_references(cell, admitted, refs, logits, inputs)
        except (ValueError, KeyError):
            checks.append("post_forward_reject_" + label)
        else:
            raise AssertionError("post-forward caller accepted " + label)
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
    direct = [Path(__file__), Path(old.__file__), Path(whole.__file__), Path(car.__file__),
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
    command = ["python", "-B", "-m", "probes.training_set_completion.recurrence_header_feedback_components.run"]
    expected_raw = 2 * a["measured_predecessor"]["raw_bytes"] + 2 * 1024 * 1024
    require(expected_raw < a["artifact_planning_bytes"] and
            a["planning_outer_seconds"] > 0, "axis-hash artifact forecast exceeds envelope")
    packet = {"status": "cpu_qualified_before_model", "protocol": bind(PROTOCOL),
              "admission": bind(ADMISSION), "repair": bind(REPAIR),
              "failed_preflight": bind(UNIT / "supporting/attempt-001-preflight.json"),
              "failed_receipt": bind(OUT1 / "receipt.json"),
              "failed_outer": bind(OUT1 / "outer.json"),
              "producer": bind(Path(__file__)),
              "source_identity": sr["identity"], "input_identity": sr["input_identity"],
              "request_ids": list(batch.request_ids), "pad_id": pad,
              "pixel_elements": int(batch.inputs["pixel_values"].numel()),
              "shapes": {"full": [4, END], "prefill": [4, WIDTH], "suffix": [4, 4],
                         "suffix_mask": [4, 1, 4, END]},
              "full_inputs": {o: {k: car.base.tensor_hash(v) for k, v in full.items()
                                   if isinstance(v, torch.Tensor)} for o, full in fulls.items()},
              "checks": checks, "source_captures": captures,
              "forecast_outer_seconds": a["planning_outer_seconds"],
              "prior_sequence_gpu_hours": json.loads(REPAIR.read_text())["accounting"]
                  ["prior_sequence_gpu_hours_for_attempt002"],
              "failed_attempt_outer_seconds": 109.01016633212566,
              "artifact_forecast_bytes_with_axis_hashes": expected_raw,
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
            p["repair"] == bind(REPAIR) and
            p["failed_preflight"] == bind(UNIT / "supporting/attempt-001-preflight.json") and
            p["failed_receipt"] == bind(OUT1 / "receipt.json") and
            p["failed_outer"] == bind(OUT1 / "outer.json") and
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
               "admission": bind(ADMISSION), "repair": bind(REPAIR),
               "failed_receipt": bind(OUT1 / "receipt.json"),
               "failed_outer": bind(OUT1 / "outer.json"),
               "preflight": bind(PREFLIGHT),
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
            mode = ("anchor" if cell["kind"] in ("anchor", "native") else
                    cell["kind"] if kind == "suffix" else kind)
            donor = (cell["older_K_origin"] if mode in ("joint", "K") else
                     cell["older_V_origin"] if mode == "V" else base)
            expected_mask = (car.expected_suffix_mask(fulls[base]) if kind == "suffix"
                             else car.base.native_4d(inputs["attention_mask"]))
            active.clear()
            active.update(name=name, kind=kind, mode=mode, base=base, donor=donor,
                          clamp=cell["clamp"], current_mode=cell["current_mode"],
                          clamped_axes=cell["clamped_axes"],
                          capture=cell["capture_native_header"],
                          input=inputs, media=bool(cell["vision"]), cache=cache,
                          expected_mask=expected_mask, attn=[], after=[],
                          rotary_values=[], rotary_hashes={},
                          qkv=[{} for _ in range(LAYERS)],
                          qkv_before=[{} for _ in range(LAYERS)],
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
                        "current_mode": cell["current_mode"],
                        "clamped_axes": cell["clamped_axes"],
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
                      "kind": cell["kind"], "reference_cell": cell["reference_cell"],
                      "clamp": cell["clamp"],
                      "current_mode": cell["current_mode"],
                      "clamped_axes": cell["clamped_axes"],
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
                    (index < 18 or all(n in vectors for n in NAMES[:18])),
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
                    accepted_capture = torch.load(a["native_capture_references"][base]["path"],
                                                  map_location="cpu", weights_only=True)
                    require(accepted_capture["base"] == base and
                            len(accepted_capture["layers"]) == LAYERS and
                            all(torch.equal(captures[base][layer][axis].cpu(),
                                            accepted_capture["layers"][layer][axis])
                                for layer in range(LAYERS) for axis in ("Q", "K", "V")),
                            f"{name} native Q/K/V capture differs from accepted")
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
                if cell["kind"] == "native":
                    error = float((logits - anchor).abs().max())
                    require(error <= TOL, f"{name} partial/whole native identity mismatch: {error}")
                    receipt["cells"][-1]["native_axis_identity_max_abs"] = error
            receipt["cells"][-1].update(post_forward_references(
                cell, cell, a, logits,
                json.loads(Path(receipt["cells"][-1]["input"]["path"]).read_text())))
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
               "probes.training_set_completion.recurrence_header_feedback_components.run", "gpu"]
    began, unix = time.monotonic(), time.time()
    with (OUT / "stdout.log").open("x") as stdout, (OUT / "stderr.log").open("x") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        code = child.wait()
    outer = {"command": command, "started_unix": unix,
             "admission": bind(ADMISSION), "repair": bind(REPAIR),
             "outer_seconds": time.monotonic() - began,
             "child_pid": child.pid, "returncode": code, "terminal": True,
             "stdout": bind(OUT / "stdout.log"), "stderr": bind(OUT / "stderr.log")}
    write_new(OUT / "outer.json", outer)
    print(json.dumps(outer))
    require(code == 0, "terminal child failure; no automatic retry")


def reduce_vectors(vectors):
    """Frozen full-vocabulary FP64 TV for both axis paths."""
    probabilities = {n: torch.softmax(v[TARGET].double(), dim=-1)
                     for n, v in vectors.items() if not n.startswith("prefill_")}
    def tv(x, y):
        return float((probabilities[x] - probabilities[y]).abs().sum() / 2)
    def below(value, threshold):
        if value is None or abs(value - threshold) <= 1e-6:
            return None
        return value < threshold
    def both(*values):
        return False if False in values else (True if all(v is True for v in values) else None)
    bases, descriptive = {}, {}
    for base, donor in (("AF", "FF"), ("FF", "AF")):
        native, joint = "anchor_" + base, f"joint_{base}_from_{donor}"
        all_joint = f"all_joint_{base}_from_{donor}"
        distance = tv(native, joint)
        informative = distance > 1e-6
        axes = {}
        for label, prefix in (("Q", "Q"), ("KV", "KV")):
            n, j = f"{prefix}_native_{base}", f"{prefix}_joint_{base}_from_{donor}"
            displacement_tv, endpoint_tv = tv(j, n), tv(j, joint)
            t = displacement_tv / distance if informative else None
            e = endpoint_tv / distance if informative else None
            axes[label] = {
                "native": n, "joint": j, "TV_joint_to_own_native": displacement_tv,
                "TV_joint_to_adaptive_joint": endpoint_tv,
                "TV_joint_to_all_joint": tv(j, all_joint),
                "TV_native_to_adaptive_native": tv(n, native),
                "t": t, "e": e, "collapse": below(t, .2), "retain": below(e, .5)}
        q, kv = axes["Q"], axes["KV"]
        predicates = {
            "query_adaptation": both(q["collapse"], kv["retain"]),
            "kv_adaptation": both(kv["collapse"], q["retain"]),
            "both_collapse": both(q["collapse"], kv["collapse"]),
            "both_retain": both(q["retain"], kv["retain"])}
        if not informative or any(x is None for x in predicates.values()):
            category = "numerical_or_denominator_HOLD"
        elif predicates["query_adaptation"]:
            category = "query_adaptation"
        elif predicates["kv_adaptation"]:
            category = "kv_adaptation"
        elif predicates["both_collapse"]:
            category = "both_collapse"
        elif predicates["both_retain"]:
            category = "both_retain"
        else:
            category = "mixed_or_changed"
        bases[base] = {"adaptive_native": native, "adaptive_joint": joint,
                       "adaptive_D": distance, "all_clamp_joint": all_joint,
                       "all_clamp_joint_to_native_TV": tv(all_joint, "all_native_" + base),
                       "axes": axes, "predicates": predicates, "category": category}
    shared = {}
    for key in ("query_adaptation", "kv_adaptation", "both_collapse", "both_retain"):
        values = [bases[b]["predicates"][key] for b in ("AF", "FF")]
        shared[key] = both(*values)
    if shared["query_adaptation"] is True:
        category = "query_adaptation_primary_pass"
    elif shared["kv_adaptation"] is True:
        category = "kv_adaptation_comparator_pass"
    elif shared["both_collapse"] is True:
        category = "shared_both_collapse"
    elif shared["both_retain"] is True:
        category = "shared_both_retain"
    elif any(x is None for x in shared.values()):
        category = "numerical_or_denominator_HOLD"
    else:
        category = "mixed_or_changed"
    for name, z4 in vectors.items():
        if name.startswith(("full_", "prefill_")):
            continue
        z = z4[TARGET].double()
        p = probabilities[name]
        top = torch.topk(z, 2)
        descriptive[name] = {
            "winner": int(top.indices[0]), "runner": int(top.indices[1]),
            "gap": float(top.values[0] - top.values[1]),
            "z_151671_minus_151670": float(z[151671] - z[151670]),
            "fixed_tokens": {str(j): {
                "probability": float(p[j]), "rank": int((z > z[j]).sum()) + 1}
                for j in (151670, 151671)}}
    return {"bases": bases, "shared": shared, "shared_outcome": category,
            "descriptive_cells": descriptive}

def readback():
    a, old_a, p = checked()
    require(not (OUT / "readback.json").exists() and
            not (UNIT / "candidate-results.md").exists(),
            "candidate readback already exists")
    receipt = json.loads((OUT / "receipt.json").read_text())
    outer = json.loads((OUT / "outer.json").read_text())
    require(receipt["status"] == "candidate_raw_complete" and
            receipt["admission"] == bind(ADMISSION) and
            receipt["repair"] == bind(REPAIR) and
            receipt["failed_receipt"] == bind(OUT1 / "receipt.json") and
            receipt["failed_outer"] == bind(OUT1 / "outer.json") and
            receipt["preflight"] == bind(PREFLIGHT) and
            receipt["producer"] == bind(Path(__file__)) and
            receipt["counts"] == {"model_forwards": 22,
                                   "vision_forwards": 4,
                                   "generated_tokens": 0} and
            outer["returncode"] == 0 and outer["terminal"] and
            outer["admission"] == bind(ADMISSION) and outer["repair"] == bind(REPAIR) and
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
        accepted_capture = torch.load(a["native_capture_references"][base]["path"],
                                      map_location="cpu", weights_only=True)
        require(accepted_capture["base"] == base and
                all(torch.equal(saved["layers"][i][axis],
                                accepted_capture["layers"][i][axis])
                    for i in range(LAYERS) for axis in ("Q", "K", "V")),
                "cold own-base native capture differs from accepted")
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
                consumer["current_mode"] == admitted["current_mode"] and
                consumer["clamped_axes"] == admitted["clamped_axes"] and
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
            mode = ("anchor" if admitted["kind"] in ("anchor", "native") else
                    admitted["kind"])
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
                for axis in ("Q", "K", "V"):
                    evidence = qkv[axis]
                    selected = axis in admitted["clamped_axes"]
                    axis_record_guard(evidence, selected=selected,
                                      capture=admitted["capture_native_header"],
                                      native_hash=car.base.tensor_hash(
                                          captures[base][layer][axis]))
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
                        "clamp": admitted["clamp"],
                        "current_mode": admitted["current_mode"],
                        "clamped_axes": admitted["clamped_axes"],
                        "input": cell["input"],
                        "vector": cell["vector"], "consumer": cell["consumer"],
                        "headouts": cell["headouts"]})
    for i, name in enumerate(NAMES):
        post_forward_references(receipt["cells"][i], a["cells"][i], a,
            vectors[name], json.loads(Path(records[i]["input"]["path"]).read_text()))
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
                    *(float((vectors[prefix + base] - anchor).abs().max())
                      for prefix in ("all_native_", "Q_native_", "KV_native_"))) <= TOL,
                "cold full/cache/sham/current-axis native all-four gate changed")
    for name in NAMES[6:]:
        base = a["cells"][NAMES.index(name)]["base_origin"]
        require(max(float((vectors[name][j] - vectors["anchor_" + base][j]).abs().max())
                    for j in (0, 1, 3)) <= TOL and
                [x["companion_suffix"] for x in consumers[name]["after_suffix"]] ==
                [x["companion_suffix"] for x in
                 consumers["anchor_" + base]["after_suffix"]] and
                all(consumers[name]["qkv"][layer][axis]["after_companions"] ==
                    consumers["anchor_" + base]["qkv"][layer][axis]["after_companions"]
                    for layer in range(LAYERS) for axis in ("Q", "K", "V")),
                f"cold {name} companion suffix Q/K/V/logits changed")
    reduced = reduce_vectors(vectors)
    repair = json.loads(REPAIR.read_text())
    failed_cost = repair["accounting"]["failed_attempt_outer_seconds"]
    prior = repair["accounting"]["prior_sequence_gpu_hours_for_attempt002"]
    accounting = {"attempt001": {"model_forwards": 13, "vision_forwards": 4,
                                 "generated_tokens": 0, "outer_seconds": failed_cost,
                                 "status": "technical_invalid"},
                  "attempt002": {**receipt["counts"], "outer_seconds": outer["outer_seconds"],
                                 "status": "candidate"},
                  "combined_model_forwards": 13 + receipt["counts"]["model_forwards"],
                  "combined_vision_forwards": 4 + receipt["counts"]["vision_forwards"],
                  "combined_generated_tokens": 0,
                  "combined_outer_seconds": failed_cost + outer["outer_seconds"],
                  "prior_sequence_gpu_hours_before_attempt002": prior,
                  "charged_sequence_gpu_hours": prior + outer["outer_seconds"] / 3600}
    result = {"status": "candidate_cold_readback_passed", "protocol": bind(PROTOCOL),
              "admission": bind(ADMISSION), "repair": bind(REPAIR),
              "failed_receipt": bind(OUT1 / "receipt.json"),
              "failed_outer": bind(OUT1 / "outer.json"),
              "preflight": bind(PREFLIGHT),
              "receipt": bind(OUT / "receipt.json"), "outer": bind(OUT / "outer.json"),
              "prefill_origins": bind(OUT / "prefill-origins.json"),
              "captures": {base: receipt["cells"][NAMES.index("anchor_" + base)]
                           ["native_header_capture"] for base in ("AF", "FF")},
              "cells": records, "counts": receipt["counts"],
              "accounting": accounting, "outcomes": reduced,
              "model_loads": 0, "model_forwards": 0,
              "vision_forwards": 0, "cuda_calls": 0,
              "gpu_seconds_added_by_readback": 0,
              "limit": "Saved-headout cold arithmetic is not a second full-SDPA replay."}
    write_new(OUT / "readback.json", result)
    lines = ["# Current-header Q versus K/V candidate", "",
             "Status: **candidate cold readback passed; lead acceptance pending.**", "",
             f"Protocol SHA `{SHAS[PROTOCOL]}`; admission SHA `{SHAS[ADMISSION]}`; "
             f"producer SHA `{bind(Path(__file__))['sha256']}`.",
             f"Receipt SHA `{result['receipt']['sha256']}`; cold readback SHA "
             f"`{bind(OUT/'readback.json')['sha256']}`. Original refined-03 four requests, "
             "target2 train351017; AF/FF common header replayed, zero generated tokens.", "",
             "## Frozen FP64 full-vocabulary decision", "",
             f"Shared category **{reduced['shared_outcome']}**. Primary both-base query-adaptation "
             f"signature **{reduced['shared']['query_adaptation']}**; symmetric K/V comparator "
             f"**{reduced['shared']['kv_adaptation']}**; both-collapse "
             f"**{reduced['shared']['both_collapse']}**; both-retain "
             f"**{reduced['shared']['both_retain']}**.", "",
             "| Base | Adaptive D | t_Q | e_Q | t_KV | e_KV | Category |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for base in ("AF", "FF"):
        row = reduced["bases"][base]
        fmt = lambda x: "undefined" if x is None else f"{x:.12g}"
        lines.append(f"| {base} | {row['adaptive_D']:.12g} | "
                     f"{fmt(row['axes']['Q']['t'])} | {fmt(row['axes']['Q']['e'])} | "
                     f"{fmt(row['axes']['KV']['t'])} | {fmt(row['axes']['KV']['e'])} | "
                     f"{row['category']} |")
    lines += ["", "All raw TVs and partial-joint TVs to the same-base all-clamped joint, "
              "full-vocabulary winners/runners/gaps, and fixed token probabilities/ranks "
              "are in the cold readback. Categorical winners do not affect the criterion.", "",
              "## Technical qualification and cost", "",
              "Calls1–18 passed fresh adaptive, whole-clamp bridge and separately executed "
              "partial-native identities before all four fixed treatments. Both own-base "
              "native Q/K/V captures exactly reproduced the accepted tensors at all28 layers "
              "and four positions. All22 raw vectors/inputs, selected older K/V donors, "
              "base latest/prehistory/companion complements, native masks/rotary, actual "
              "pre- and post-actuation Q/K/V, selected/free axes, appended K/V, scale-aware "
              "FP64 pre-o_proj reconstruction and finally restoration were checked. "
              "Cold readback recomputed saved-headout arithmetic and full-vocabulary TVs "
              "in a separate CPU process; it was not a second full SDPA replay.", "",
              f"Actual {receipt['counts']['model_forwards']} model / "
              f"{receipt['counts']['vision_forwards']} vision / "
              f"{receipt['counts']['generated_tokens']} generated. Parent outer "
              f"{outer['outer_seconds']:.9f} s; internal {receipt['internal_seconds']:.9f} s. "
              f"Failed attempt001 13 model/4 vision/0 generated, outer "
              f"{failed_cost:.9f} s remains charged. Combined attempts "
              f"{accounting['combined_model_forwards']} model/"
              f"{accounting['combined_vision_forwards']} vision/0 generated, outer "
              f"{accounting['combined_outer_seconds']:.9f} s. Prior sequence before "
              f"attempt002 {prior:.12f} GPUh; charged cumulative "
              f"{accounting['charged_sequence_gpu_hours']:.12f} GPUh.",
              f"Peak RSS {receipt['rss_peak_kib']} KiB; GPU allocated/reserved "
              f"{receipt['gpu_peak_allocated_bytes']}/{receipt['gpu_peak_reserved_bytes']} B; "
              f"artifact bytes before receipt {receipt['artifact_bytes_before_receipt']}; "
              f"terminal child PID {outer['child_pid']}, exit {outer['returncode']}.", "",
              f"Raw attempt: `{OUT}`. Technical bounds and scientific thresholds remain "
              "frozen. F/F2 physical identity stays HOLD. The intervention tests "
              "within-forward adaptation; it does not establish a natural recurrence loop "
              "or mediation fraction. No self-acceptance or successor.", ""]
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
