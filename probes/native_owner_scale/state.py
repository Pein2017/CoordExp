"""Frozen cross-scene coordinate-carrier K-cache intervention.

This lane deliberately imports the accepted instance-state masking/transplant
machinery.  It adds only the fixed multi-scene packet, K-only norm control,
receipt binding, and cold consumer needed by the current research unit.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import torch

from probes.parallel_owner_research.instance_state import (
    EOS,
    OPENER,
    BOX_END,
    cache_slices,
    digest,
    file_hash,
    generation_inputs,
    grounding_mask,
    region_indices,
    require,
    transplant,
    write,
)
from probes.parallel_owner_research.instance_state_amplitude import (
    diagnostics,
    do_prefill,
    flatten_slices,
    norm64,
    tensor_hash,
)


RAW_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/state"
)
SOURCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-small-owner-repeat-origin/packet-repair-01.json"
)
PARENT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-parallel-owner-research/instance-state/panel.json"
)
AMPLITUDE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-parallel-owner-research/instance-state/amplitude-control/run-v1"
)
ARMS = ("native_self", "owner_a_k", "owner_b_k_to_a")
TOTAL_ACTION_LIMIT = 3084


def canonical_terminal_stop(raw_stop, suffix_length, budget):
    """Normalize native `im_end` and local `eos` without hiding early stops."""
    require(raw_stop in ("im_end", "eos", "length"), f"unknown stop reason: {raw_stop}")
    if raw_stop in ("im_end", "eos"):
        require(suffix_length <= budget, "EOS suffix exceeds remaining allowance")
        return "eos"
    require(suffix_length == budget, "branch stopped before EOS/full remaining allowance")
    return "length"


def _key_vector(slices):
    return torch.cat([keys.detach().cpu().reshape(-1).to(torch.float64) for keys, _ in slices])


def _keys_from_vector(vector, template):
    result, offset = [], 0
    for keys, values in template:
        count = keys.numel()
        rebuilt = vector[offset : offset + count].reshape(keys.shape).to(keys.dtype)
        result.append((rebuilt, values.clone()))
        offset += count
    require(offset == vector.numel(), "K-vector reconstruction mismatch")
    return result


def k_only_branch_slices(captured, arm):
    """Return one frozen K-only arm and its realized global-L2 diagnostics."""
    require(arm in ARMS, "unregistered state-transfer arm")
    native = captured["native"]
    base = _key_vector(native)
    da = _key_vector(captured["owner_a"]) - base
    db = _key_vector(captured["owner_b"]) - base
    na, nb = norm64(da), norm64(db)
    require(na > 0 and nb > 0, "zero-norm A/B K delta blocks frozen contrast")
    scale = None
    if arm == "native_self":
        result = [(keys.clone(), values.clone()) for keys, values in native]
    elif arm == "owner_a_k":
        result = [(a_keys.clone(), n_values.clone()) for (a_keys, _), (_, n_values) in zip(captured["owner_a"], native, strict=True)]
    else:
        scale = na / nb
        result = _keys_from_vector(base + db * scale, native)
        observed = norm64(_key_vector(result) - base)
        require(abs(observed - na) / na < 5e-5, "realized wrong-owner K norm departed from A")
    full_delta = flatten_slices(result) - flatten_slices(native)
    stats = {
        "arm": arm,
        "scale": scale,
        "norm_rule": "single global float64 L2 over all layers,K-only,4 coordinate positions",
        "a_k_norm": na,
        "b_k_norm": nb,
        "realized_k_norm": norm64(_key_vector(result) - base),
        "realized_full_delta": diagnostics(full_delta, native),
    }
    return result, stats


def _parameter_inventory(model):
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "data_ptr": parameter.data_ptr(),
            "version": parameter._version,
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    ]


def _reference(path):
    path = Path(path)
    value = json.loads(path.read_text())
    return {"path": str(path), "sha256": file_hash(path), "suffix_ids": value["suffix_ids"]}


def _case_behavior(case, tokenizer):
    from src.eval.native_rows import native_detection_record as native_record
    from probes.dora_owner_learning.candidate_opportunity import score

    action = case["baseline_action_ids"]
    stop = "eos" if action[-1] == EOS else "length"
    parsed = native_record(tokenizer.decode(action, skip_special_tokens=False), case["source_case"], case["golden"], stop)
    scored = score(parsed, seed=None, length=len(action), stop=stop)
    return {
        "total_tokens": len(action),
        "stop": stop,
        "annotated_gt": len(parsed["gt"]),
        "tp50": scored["50"]["tp"],
        "fn50": scored["50"]["fn"],
        "strict_repeats": scored["strict_repeats"],
        "parser_drops": scored["parser_drops"],
    }


def _render_carrier_card(case, output):
    from PIL import Image, ImageDraw

    image_path = Path(case["source_case"]["image_plan"]["image_path"])
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for name, color in (("owner_a", "red"), ("owner_b", "deepskyblue")):
        x1, y1, x2, y2 = case["boxes"][name]
        pixel = [x1 * image.width / 999, y1 * image.height / 999, x2 * image.width / 999, y2 * image.height / 999]
        draw.rectangle(pixel, outline=color, width=max(3, min(image.size) // 180))
        draw.text((pixel[0] + 4, pixel[1] + 4), name, fill=color, stroke_width=2, stroke_fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "case_id": case["case_id"],
        "original_path": str(image_path),
        "original_sha256": file_hash(image_path),
        "card_path": str(output),
        "card_sha256": file_hash(output),
        "boxes": case["boxes"],
    }


def prepare():
    """Freeze the four exposed scenes, boundaries, arms, and launch topology."""
    from tokenizers import Tokenizer

    source = json.loads(SOURCE.read_text())
    parent = json.loads(PARENT_PANEL.read_text())
    tokenizer = Tokenizer.from_file(source["config"]["model"]["base_model"] + "/tokenizer.json")
    source_cases = {case["case_id"]: case for case in source["cases"]}
    parent_cases = {case["case_id"]: copy.deepcopy(case) for case in parent["cases"]}

    # Existing visually admitted boundaries are reused byte-for-byte.
    cases = []
    for case_id, role in (("9813", "normal_finite_control"), ("417044", "known_positive_donut_loop"), ("477415", "non_donut_chair_loop")):
        case = parent_cases[case_id]
        case["role"] = role
        case["boxes"] = {name: case["boxes"][name] for name in ("owner_a", "owner_b")}
        case["regions"] = {name: case["regions"][name] for name in ("owner_a", "owner_b")}
        case["carriers"] = {"coordinates": case["carriers"]["coordinates"]}
        if case_id == "9813":
            case["visual_admission"] = "A is the seated driver on the left; B is the separate foreground woman on the right. Both are literal saved Stable50 prediction regions checked on the original photo."
        case["source_behavior"] = _case_behavior(source_cases[case_id], tokenizer)
        cases.append(case)

    # 158044 is a previously exposed finite output.  A is its third emitted
    # row (the remote in the child's hands); B is the distinct teddy on chair.
    old = source_cases["158044"]
    history_end = 27
    history = old["baseline_action_ids"][:history_end]
    require(history[-1] == BOX_END and old["baseline_action_ids"][history_end] == OPENER, "158044 boundary is not a native closed row/opener")
    prompt = old["prompt_token_ids"]
    boxes = {"owner_a": [450, 420, 500, 465], "owner_b": [549, 233, 653, 308]}
    grid = old["source_case"]["image_plan"]["observed_image_grid_thw"]
    args = (prompt, grid, tokenizer.token_to_id("<|image_pad|>"))
    regions = {"owner_a": region_indices(*args, boxes["owner_a"])}
    regions["owner_b"] = region_indices(*args, boxes["owner_b"], len(regions["owner_a"]))
    require(not (set(regions["owner_a"]) & set(regions["owner_b"])), "158044 A/B image keys overlap")
    start = max(index for index, token in enumerate(history) if token == OPENER)
    cases.append(
        {
            "case_id": "158044",
            "role": "finite_mixed_missing_repeat",
            "source_case": old["source_case"],
            "golden": old["golden"],
            "prompt_ids": prompt,
            "history_ids": history,
            "opener": OPENER,
            "expected_suffix": old["baseline_action_ids"][history_end + 1 :],
            "boxes": boxes,
            "regions": regions,
            "queries": list(range(len(prompt) + start, len(prompt) + history_end)),
            "carriers": {"coordinates": list(range(len(prompt) + history_end - 5, len(prompt) + history_end - 1))},
            "visual_admission": "A is the real handheld remote; B is the separate white teddy bear on the orange chair. Both are literal saved Stable50 prediction regions and were checked on the original photo before treatment outputs.",
            "source_behavior": _case_behavior(old, tokenizer),
        }
    )

    cases.sort(key=lambda case: int(case["case_id"]))
    expected_behavior = {
        "9813": (100, "eos", 9, 0, 0),
        "158044": (246, "eos", 4, 13, 5),
        "417044": (3084, "length", 1, 14, 291),
        "477415": (3084, "length", 2, 25, 4),
    }
    for case in cases:
        require(len(case["carriers"]["coordinates"]) == 4, "coordinate carrier changed")
        require(len(case["regions"]["owner_a"]) == len(case["regions"]["owner_b"]), "A/B mask key counts differ")
        observed = case["source_behavior"]
        frozen = expected_behavior[case["case_id"]]
        require((observed["total_tokens"], observed["stop"], observed["tp50"], observed["fn50"], observed["strict_repeats"]) == frozen, "source behavior drift")

    cards = []
    for case in cases:
        cards.append(_render_carrier_card(case, RAW_ROOT / "preparation" / "carrier-cards" / f"{case['case_id']}-a-b.jpg"))
    write(RAW_ROOT / "preparation" / "carrier-manifest.json", {"status": "direct_photo_cards", "cards": cards})

    references = {
        "native": _reference(PARENT_PANEL.parent / "panel-v1" / "417044-native.json"),
        "native_self": _reference(AMPLITUDE_ROOT / "gate" / "native_self.json"),
        "owner_a_k": _reference(AMPLITUDE_ROOT / "worker-6" / "a_k_only.json"),
    }
    require(references["native"]["suffix_ids"] == references["native_self"]["suffix_ids"], "old native/self reference mismatch")
    config = copy.deepcopy(source["config"])
    config["adapter"]["path"] = source["anchor_adapter"]
    source_files = {
        str(Path(__file__).resolve()): file_hash(__file__),
        str(Path(__file__).parents[1] / "parallel_owner_research" / "instance_state.py"): file_hash(Path(__file__).parents[1] / "parallel_owner_research" / "instance_state.py"),
        str(Path(__file__).parents[1] / "parallel_owner_research" / "instance_state_amplitude.py"): file_hash(Path(__file__).parents[1] / "parallel_owner_research" / "instance_state_amplitude.py"),
    }
    completed_gate = None
    old_packet = RAW_ROOT / "packet.json"
    gate_dir = RAW_ROOT / "gate-v1"
    if (gate_dir / "gate.json").is_file():
        require(old_packet.is_file(), "completed gate predecessor packet missing")
        old_receipt = json.loads((gate_dir / "receipt.json").read_text())
        require(old_receipt["status"] == "complete" and old_receipt["packet_sha256"] == file_hash(old_packet), "completed gate predecessor mismatch")
        completed_gate = {
            "packet": str(old_packet),
            "packet_sha256": file_hash(old_packet),
            "directory": str(gate_dir),
            "gate_sha256": file_hash(gate_dir / "gate.json"),
            "receipt_sha256": file_hash(gate_dir / "receipt.json"),
            "consumer_sha256": file_hash(gate_dir / "consumer.json"),
            "runner_sha256": file_hash(gate_dir / "runner.py"),
            "case_ids": ["417044"],
        }

    packet = {
        "schema": "native_owner_state_transfer.v1",
        "status": "frozen_continuation_after_valid_gate" if completed_gate else "frozen_before_treatment_outputs",
        "question": "Does the known A-region K-only coordinate-carrier intervention transfer across fixed loop, missing/mixed, and normal scenes while preserving native owners?",
        "claim_boundary": "Conditional fixed-history cache intervention only; no autonomous owner, architecture, ledger, or general detector claim.",
        "source_packet": str(SOURCE),
        "source_sha256": file_hash(SOURCE),
        "parent_panel": str(PARENT_PANEL),
        "parent_panel_sha256": file_hash(PARENT_PANEL),
        "carrier_manifest": str(RAW_ROOT / "preparation" / "carrier-manifest.json"),
        "carrier_manifest_sha256": file_hash(RAW_ROOT / "preparation" / "carrier-manifest.json"),
        "config": config,
        "anchor_adapter": source["anchor_adapter"],
        "cases": cases,
        "arms": list(ARMS),
        "arm_family": "coordinate positions across all text layers; K-only; same-scene B K delta globally L2-matched to A K; no V/full-KV/layer/head/amplitude sweep",
        "decode": {"temperature": 0, "top_p": 1, "repetition_penalty": 1.0, "dtype": "float32", "attention": "sdpa", "total_action_limit": TOTAL_ACTION_LIMIT},
        "gate": {"gpu": "6", "case_ids": ["417044"], "references": references},
        "workers": [{"gpu": "6", "case_ids": ["9813", "477415"]}, {"gpu": "7", "case_ids": ["158044"]}],
        "selection": {"scene_ids": [case["case_id"] for case in cases], "selected_before_treatment_outputs": True, "no_backfill": True, "evaluation_exclusion_notified": True},
        "source_files": source_files,
    }
    if completed_gate:
        packet["completed_gate"] = completed_gate
    packet_path = RAW_ROOT / ("packet-v2.json" if completed_gate else "packet.json")
    write(packet_path, packet)
    maximum_decode = sum((1 + len(ARMS)) * (TOTAL_ACTION_LIMIT - len(case["history_ids"]) - 1) for case in cases)
    return {
        "packet": str(packet_path),
        "sha256": file_hash(packet_path),
        "cases": [case["case_id"] for case in cases],
        "roles": {case["case_id"]: case["role"] for case in cases},
        "arms": list(ARMS),
        "matched_region_keys": {case["case_id"]: len(case["regions"]["owner_a"]) for case in cases},
        "maximum_decode_forwards": maximum_decode,
        "model_loads": 3,
        "maximum_vision_forwards": 16,
    }


def _load_policy(packet, receipt):
    from src.config.inference import InferConfig
    from probes.dora_owner_learning.runtime import load_policy

    qwen, identity = load_policy(InferConfig.model_validate(packet["config"]), device=torch.device("cuda:0"))
    require(identity["model_identity"]["adapter"]["adapter_path"] == packet["anchor_adapter"], "loaded adapter is not Stable50")
    require(identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "loaded attention is not SDPA")
    require(next(qwen.model.parameters()).dtype == torch.float32, "model is not FP32")
    receipt["model_forwards"] = 0
    receipt["vision_forwards"] = 0
    qwen.model.register_forward_pre_hook(lambda *_: receipt.__setitem__("model_forwards", receipt["model_forwards"] + 1))
    vision = [module for module in qwen.model.modules() if type(module).__name__ == "Qwen3VLVisionModel"]
    require(len(vision) == 1, "vision module identity mismatch")
    vision[0].register_forward_pre_hook(lambda *_: receipt.__setitem__("vision_forwards", receipt["vision_forwards"] + 1))
    return qwen, identity


def _save_capture(path, captured):
    from safetensors.torch import save_file

    payload = {
        f"{source}.layer{layer:02d}.{kind}": tensor.detach().cpu().contiguous()
        for source, slices in captured.items()
        for layer, pair in enumerate(slices)
        for kind, tensor in zip(("k", "v"), pair)
    }
    save_file(payload, str(path))


def _run_one_case(qwen, packet, case, directory, receipt):
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations, NativeGenerationPolicy

    directory.mkdir(parents=True, exist_ok=False)
    requests, _ = build_requests(qwen, packet["config"], [case["source_case"]])
    batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"), record_media_identity=True)
    require(list(batch.prompt_token_ids[0]) == case["prompt_ids"], "native prompt mismatch")
    history = case["history_ids"]
    budget = TOTAL_ACTION_LIMIT - len(history) - 1
    require(budget > 0 and budget == packet["decode"]["total_action_limit"] - len(history) - 1, "total action budget drift")
    before_parameters = _parameter_inventory(qwen.model)
    before_vision = receipt["vision_forwards"]

    with torch.inference_mode():
        native = generate_continuations(
            qwen.model,
            batch,
            extensions=[history + [case["opener"]]],
            budgets=[budget],
            eos_token_id=EOS,
            pad_token_id=qwen.tokenizer.pad_token_id,
            policy=NativeGenerationPolicy(),
        )[0]
        native_suffix = list(native.token_ids)
        require(native_suffix == case["expected_suffix"][:budget], "fresh native suffix differs from frozen Stable50 output")
        native_stop = canonical_terminal_stop(native.stop_reason, len(native_suffix), budget)
        write(
            directory / "native.json",
            {
                "case_id": case["case_id"],
                "arm": "native",
                "prefix_ids": history + [case["opener"]],
                "suffix_ids": native_suffix,
                "stop": native_stop,
                "raw_stop_reason": native.stop_reason,
                "text": qwen.tokenizer.decode(history + [case["opener"]] + native_suffix, skip_special_tokens=False),
                "budget": budget,
            },
        )

        ids = torch.tensor([case["prompt_ids"] + history], device="cuda:0")
        baseline_prefill, positions = do_prefill(qwen, batch, ids)
        baseline = baseline_prefill.past_key_values
        rope_delta = qwen.model.model.rope_deltas.clone()
        del baseline_prefill
        require(baseline.get_seq_length() == ids.shape[1], "prefill cache length mismatch")
        captured = {"native": [(key.cpu(), value.cpu()) for key, value in cache_slices(baseline, case["carriers"]["coordinates"])]}
        masks = {}
        for source in ("owner_a", "owner_b"):
            masks[source] = {}
            with grounding_mask(qwen.model, case["queries"], case["regions"][source], ids.shape[1], masks[source]):
                donor, donor_positions = do_prefill(qwen, batch, ids)
            require(donor_positions == positions, "masked donor MRoPE source positions changed")
            require(torch.equal(qwen.model.model.rope_deltas, rope_delta), "masked donor MRoPE delta changed")
            captured[source] = [(key.cpu(), value.cpu()) for key, value in cache_slices(donor.past_key_values, case["carriers"]["coordinates"])]
            del donor
        _save_capture(directory / "donor_slices.safetensors", captured)
        plans = {arm: k_only_branch_slices(captured, arm)[1] for arm in ARMS}
        write(directory / "frozen_branches.json", plans)
        write(directory / "masks.json", masks)

        full_ids = torch.cat([ids, torch.tensor([[case["opener"]]], device=ids.device)], dim=1)
        arm_refs = []
        for arm in ARMS:
            chosen, stats = k_only_branch_slices(captured, arm)
            require(stats == plans[arm], "frozen branch changed between capture and decode")
            cache = copy.deepcopy(baseline)
            change = transplant(cache, [(key.to(ids.device), value.to(ids.device)) for key, value in chosen], case["carriers"]["coordinates"])
            qwen.model.model.rope_deltas = rope_delta.clone()
            consumed = []

            def record_first(_module, _args, kwargs):
                consumed.append(
                    {
                        "length": kwargs["input_ids"].shape[1],
                        "cache_length": kwargs["past_key_values"].get_seq_length(),
                        "cache_position": kwargs["cache_position"].tolist(),
                    }
                )

            hook = qwen.model.register_forward_pre_hook(record_first, with_kwargs=True)
            decode_vision = receipt["vision_forwards"]
            try:
                generated = qwen.model.generate(
                    **generation_inputs(batch, full_ids),
                    past_key_values=cache,
                    cache_position=torch.tensor([ids.shape[1]], device=ids.device),
                    max_new_tokens=budget,
                    do_sample=False,
                    repetition_penalty=1.0,
                    eos_token_id=EOS,
                    pad_token_id=qwen.tokenizer.pad_token_id,
                    return_dict_in_generate=False,
                    output_scores=False,
                    output_logits=False,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            finally:
                hook.remove()
            require(receipt["vision_forwards"] == decode_vision, "cached decode repeated vision")
            require(consumed and consumed[0] == {"length": 1, "cache_length": ids.shape[1], "cache_position": [ids.shape[1]]}, "cached opener/source position mismatch")
            suffix = generated[0, full_ids.shape[1] :].tolist()
            raw_stop = "eos" if suffix[-1] == EOS else "length"
            stop = canonical_terminal_stop(raw_stop, len(suffix), budget)
            if arm == "native_self":
                require(change["changed_scalars"] == 0, "self transplant changed cache")
                require(suffix == native_suffix, "native/self full-suffix parity failed")
            cell = {
                "case_id": case["case_id"],
                "arm": arm,
                "prefix_ids": history + [case["opener"]],
                "suffix_ids": suffix,
                "stop": stop,
                "raw_stop_reason": raw_stop,
                "text": qwen.tokenizer.decode(history + [case["opener"]] + suffix, skip_special_tokens=False),
                "budget": budget,
                "delta_stats": stats,
                "cache_change": change,
                "mrope_positions": positions,
                "rope_delta_sha256": tensor_hash(rope_delta),
                "first_consumption": consumed[0],
                "decode_forwards": len(consumed),
            }
            path = directory / f"{arm}.json"
            write(path, cell)
            arm_refs.append({"arm": arm, "path": str(path), "sha256": file_hash(path)})
            del cache, generated

    require(_parameter_inventory(qwen.model) == before_parameters, "model parameter inventory/version changed")
    case_receipt = {
        "status": "complete",
        "case_id": case["case_id"],
        "role": case["role"],
        "budget": budget,
        "native_sha256": file_hash(directory / "native.json"),
        "capture_sha256": file_hash(directory / "donor_slices.safetensors"),
        "branches_sha256": file_hash(directory / "frozen_branches.json"),
        "masks_sha256": file_hash(directory / "masks.json"),
        "arms": arm_refs,
        "parameter_inventory_sha256": digest(before_parameters),
        "parameter_count": len(before_parameters),
        "vision_forwards": receipt["vision_forwards"] - before_vision,
    }
    write(directory / "receipt.json", case_receipt)
    return {"case_id": case["case_id"], "receipt_path": str(directory / "receipt.json"), "receipt_sha256": file_hash(directory / "receipt.json")}


def run_stage(packet_path, out_dir, case_ids):
    packet_path, out = Path(packet_path), Path(out_dir)
    packet = json.loads(packet_path.read_text())
    require(file_hash(packet["source_packet"]) == packet["source_sha256"], "source packet changed")
    require(file_hash(packet["parent_panel"]) == packet["parent_panel_sha256"], "parent instance-state panel changed")
    require(file_hash(packet["carrier_manifest"]) == packet["carrier_manifest_sha256"], "carrier photo manifest changed")
    for path, expected in packet["source_files"].items():
        require(file_hash(path) == expected, f"executed source changed: {path}")
    require(torch.cuda.device_count() == 1, "stage requires exactly one visible GPU")
    selected = [case for case in packet["cases"] if case["case_id"] in case_ids]
    require(len(case_ids) == len(set(case_ids)), "duplicate stage case")
    require([case["case_id"] for case in selected] == [case["case_id"] for case in packet["cases"] if case["case_id"] in set(case_ids)], "case order changed")
    require(set(case_ids) == {case["case_id"] for case in selected}, "unknown/duplicate stage case")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "runner.py")
    shutil.copyfile(packet_path, out / "packet.json")
    receipt = {
        "status": "running",
        "packet": str(packet_path),
        "packet_sha256": file_hash(packet_path),
        "runner_sha256": file_hash(__file__),
        "dependency_sha256": packet["source_files"],
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "case_ids": list(case_ids),
        "cases": [],
    }
    started = time.monotonic()
    write(out / "receipt.json", receipt)
    try:
        qwen, identity = _load_policy(packet, receipt)
        write(out / "model.json", identity)
        for case in selected:
            receipt["cases"].append(_run_one_case(qwen, packet, case, out / "cases" / case["case_id"], receipt))
            write(out / "receipt.json", receipt)
        receipt["status"] = "complete"
    except BaseException:
        receipt.update(status="technical_invalid", traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(
            wall_seconds=time.monotonic() - started,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        write(out / "receipt.json", receipt)
    return receipt


def _score_cell(tokenizer, case, cell, native):
    from src.eval.native_rows import native_detection_record as native_record
    from probes.dora_owner_learning.candidate_opportunity import score

    prefix = case["history_ids"]
    action = cell["prefix_ids"] + cell["suffix_ids"]
    require(cell["prefix_ids"] == prefix + [case["opener"]], "cell prefix changed")
    require(tokenizer.decode(action, skip_special_tokens=False) == cell["text"], "cell token/text mismatch")
    full = native_record(cell["text"], case["source_case"], case["golden"], cell["stop"])
    full_score = score(full, seed=None, length=len(action), stop=cell["stop"])
    free_action = [case["opener"]] + cell["suffix_ids"]
    free_text = tokenizer.decode(free_action, skip_special_tokens=False)
    free = native_record(free_text, case["source_case"], case["golden"], cell["stop"])
    free_score = score(free, seed=None, length=len(free_action), stop=cell["stop"])
    result = {
        "case_id": case["case_id"],
        "role": case["role"],
        "arm": cell["arm"],
        "full_score": full_score,
        "free_score": free_score,
        "full_parsed": full,
        "free_parsed": free,
        "raw_row_starts": action.count(case["opener"]),
        "free_raw_row_starts": free_action.count(case["opener"]),
        "total_action_tokens": len(action),
        "free_tokens": len(cell["suffix_ids"]),
        "stop": cell["stop"],
        "delta_stats": cell.get("delta_stats"),
        "suffix_equal_native": cell["suffix_ids"] == native["suffix_ids"],
    }
    return result


def consume_stage(packet_path, out_dir):
    from tokenizers import Tokenizer
    from src.eval.native_rows import native_detection_record as native_record
    from probes.dora_owner_learning.candidate_opportunity import score

    packet_path, out = Path(packet_path), Path(out_dir)
    packet = json.loads(packet_path.read_text())
    receipt = json.loads((out / "receipt.json").read_text())
    require(receipt["status"] == "complete" and receipt["packet_sha256"] == file_hash(packet_path), "stage incomplete or packet mismatch")
    tokenizer = Tokenizer.from_file(packet["config"]["model"]["base_model"] + "/tokenizer.json")
    cases = {case["case_id"]: case for case in packet["cases"]}
    records = []
    for case_ref in receipt["cases"]:
        directory = out / "cases" / case_ref["case_id"]
        require(file_hash(directory / "receipt.json") == case_ref["receipt_sha256"], "case receipt changed")
        case_receipt = json.loads((directory / "receipt.json").read_text())
        require(case_receipt["status"] == "complete", "case technically invalid")
        require(case_receipt["capture_sha256"] == file_hash(directory / "donor_slices.safetensors"), "capture bytes changed")
        case = cases[case_ref["case_id"]]
        native = json.loads((directory / "native.json").read_text())
        require(file_hash(directory / "native.json") == case_receipt["native_sha256"], "native output changed")
        native_cell = dict(native, arm="native")
        native_record_result = _score_cell(tokenizer, case, native_cell, native)
        records.append(native_record_result)
        native_full, native_free = native_record_result["full_score"], native_record_result["free_score"]
        seen = []
        for arm_ref in case_receipt["arms"]:
            path = Path(arm_ref["path"])
            require(file_hash(path) == arm_ref["sha256"], "arm output bytes changed")
            cell = json.loads(path.read_text())
            result = _score_cell(tokenizer, case, cell, native)
            for surface, baseline in (("full", native_full), ("free", native_free)):
                score_value = result[f"{surface}_score"]
                result[f"{surface}_gained"] = {str(threshold): sorted(set(score_value[str(threshold)]["owners"]) - set(baseline[str(threshold)]["owners"])) for threshold in (50, 60, 80)}
                result[f"{surface}_lost"] = {str(threshold): sorted(set(baseline[str(threshold)]["owners"]) - set(score_value[str(threshold)]["owners"])) for threshold in (50, 60, 80)}
            records.append(result)
            seen.append(cell["arm"])
        require(tuple(seen) == ARMS, "arm order/coverage changed")

    result = {
        "status": "cold_consumer_verified",
        "scope": "Conditional free suffix; supplied history owners receive no autonomous recovery credit. GT matches and FP are annotation-relative; unmatched physical owners require direct review.",
        "packet_sha256": file_hash(packet_path),
        "receipt_sha256": file_hash(out / "receipt.json"),
        "records": records,
    }
    write(out / "consumer.json", result)
    return {"consumer": str(out / "consumer.json"), "sha256": file_hash(out / "consumer.json"), "cases": len(receipt["cases"]), "records": len(records)}


def gate(packet_path, out_dir):
    packet = json.loads(Path(packet_path).read_text())
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == packet["gate"]["gpu"], "gate must use frozen physical GPU")
    run_stage(packet_path, out_dir, packet["gate"]["case_ids"])
    result = consume_stage(packet_path, out_dir)
    directory = Path(out_dir) / "cases" / packet["gate"]["case_ids"][0]
    native = json.loads((directory / "native.json").read_text())
    self_cell = json.loads((directory / "native_self.json").read_text())
    a_cell = json.loads((directory / "owner_a_k.json").read_text())
    refs = packet["gate"]["references"]
    for reference in refs.values():
        require(file_hash(reference["path"]) == reference["sha256"], "gate reference changed")
    require(native["suffix_ids"] == refs["native"]["suffix_ids"], "fresh native gate reference failed")
    require(self_cell["suffix_ids"] == refs["native_self"]["suffix_ids"], "self gate reference failed")
    require(a_cell["suffix_ids"] == refs["owner_a_k"]["suffix_ids"], "A-K gate reference failed")
    write(
        Path(out_dir) / "gate.json",
        {
            "status": "reference_consumer_unchanged_self_pass",
            "packet_sha256": file_hash(packet_path),
            "consumer_sha256": result["sha256"],
            "references": {name: {"path": value["path"], "sha256": value["sha256"]} for name, value in refs.items()},
        },
    )
    return {**result, "gate": str(Path(out_dir) / "gate.json"), "gate_sha256": file_hash(Path(out_dir) / "gate.json")}


def launch(packet_path, out_dir, gate_dir):
    packet_path, out, gate_dir = Path(packet_path), Path(out_dir), Path(gate_dir)
    packet = json.loads(packet_path.read_text())
    require("completed_gate" in packet, "versioned completed-gate binding absent")
    completed_gate = packet["completed_gate"]
    require(Path(completed_gate["directory"]) == gate_dir, "gate directory differs from packet")
    require(file_hash(completed_gate["packet"]) == completed_gate["packet_sha256"], "predecessor packet changed")
    gate_record = json.loads((gate_dir / "gate.json").read_text())
    require(gate_record["status"] == "reference_consumer_unchanged_self_pass", "root-admitted gate artifact absent")
    require(gate_record["packet_sha256"] == completed_gate["packet_sha256"], "gate artifact predecessor packet differs")
    require(file_hash(gate_dir / "gate.json") == completed_gate["gate_sha256"], "gate artifact changed")
    require(file_hash(gate_dir / "receipt.json") == completed_gate["receipt_sha256"], "gate receipt changed")
    require(file_hash(gate_dir / "consumer.json") == completed_gate["consumer_sha256"], "gate consumer changed")
    acceptance_path = gate_dir / "root-launch-acceptance.json"
    require(acceptance_path.is_file(), "root launch acceptance receipt absent")
    acceptance = json.loads(acceptance_path.read_text())
    require(
        acceptance == {
            "status": "root_accepted_persistence_consumer_unchanged_self_gate",
            "packet_sha256": file_hash(packet_path),
            "gate_sha256": file_hash(gate_dir / "gate.json"),
        },
        "root launch acceptance does not bind this exact gate",
    )
    require(json.loads((gate_dir / "receipt.json").read_text())["packet_sha256"] == completed_gate["packet_sha256"], "gate packet differs")
    out.mkdir(parents=True, exist_ok=False)
    launch_path = out / "launch.json"
    write(launch_path, {"status": "started", "packet_sha256": file_hash(packet_path), "gate_dir": str(gate_dir), "gate_sha256": file_hash(gate_dir / "gate.json"), "root_acceptance_sha256": file_hash(acceptance_path), "workers": []})

    running = []
    for worker in packet["workers"]:
        name = "worker-" + worker["gpu"]
        command = [sys.executable, "-m", "probes.native_owner_scale.state", "worker", "--packet", str(packet_path), "--out-dir", str(out / name), "--case-ids", *worker["case_ids"]]
        log = (out / f"{name}.log").open("w")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=worker["gpu"]))
        running.append((name, worker, command, process, log))
    completed, errors = [], []
    try:
        for name, worker, command, process, log in running:
            code = process.wait()
            log.close()
            (out / f"{name}.exit").write_text(str(code) + "\n")
            record = {"name": name, "gpu": worker["gpu"], "case_ids": worker["case_ids"], "command": command, "exit": code}
            if (out / name / "receipt.json").is_file():
                record["receipt_sha256"] = file_hash(out / name / "receipt.json")
            completed.append(record)
            if code != 0:
                errors.append(f"{name} exited {code}")
        require(not errors, "; ".join(errors))
        write(launch_path, {"status": "complete", "packet_sha256": file_hash(packet_path), "gate_dir": str(gate_dir), "gate_sha256": file_hash(gate_dir / "gate.json"), "root_acceptance_sha256": file_hash(acceptance_path), "workers": completed})
    except BaseException:
        for _, _, _, process, log in running:
            if process.poll() is None:
                process.wait()
            if not log.closed:
                log.close()
        write(launch_path, {"status": "technical_invalid", "packet_sha256": file_hash(packet_path), "gate_dir": str(gate_dir), "workers": completed, "errors": errors, "traceback": traceback.format_exc()})
        raise
    return consume_panel(packet_path, out, gate_dir)


def consume_panel(packet_path, out_dir, gate_dir):
    packet_path, out, gate_dir = Path(packet_path), Path(out_dir), Path(gate_dir)
    packet = json.loads(packet_path.read_text())
    launch_record = json.loads((out / "launch.json").read_text())
    require(launch_record["status"] == "complete" and launch_record["packet_sha256"] == file_hash(packet_path), "panel launch incomplete")
    completed_gate = packet["completed_gate"]
    require(Path(completed_gate["directory"]) == gate_dir, "consumer gate directory differs")
    require(file_hash(gate_dir / "gate.json") == completed_gate["gate_sha256"], "consumer gate marker changed")
    require(file_hash(gate_dir / "receipt.json") == completed_gate["receipt_sha256"], "consumer gate receipt changed")
    require(file_hash(gate_dir / "consumer.json") == completed_gate["consumer_sha256"], "consumer gate output changed")
    stage_dirs = [out / worker["name"] for worker in launch_record["workers"]]
    stage_consumers = []
    gate_consumer = json.loads((gate_dir / "consumer.json").read_text())
    require(gate_consumer["status"] == "cold_consumer_verified", "frozen gate consumer status changed")
    all_records = list(gate_consumer["records"])
    seen_cases = list(completed_gate["case_ids"])
    stage_consumers.append({"path": str(gate_dir / "consumer.json"), "sha256": completed_gate["consumer_sha256"], "receipt_sha256": completed_gate["receipt_sha256"], "predecessor_packet_sha256": completed_gate["packet_sha256"]})
    for directory in stage_dirs:
        result = consume_stage(packet_path, directory)
        consumer = json.loads(Path(result["consumer"]).read_text())
        all_records.extend(consumer["records"])
        receipt = json.loads((directory / "receipt.json").read_text())
        seen_cases.extend(receipt["case_ids"])
        stage_consumers.append({"path": result["consumer"], "sha256": result["sha256"], "receipt_sha256": file_hash(directory / "receipt.json")})
    require(sorted(seen_cases, key=int) == sorted((case["case_id"] for case in packet["cases"]), key=int), "panel case coverage changed")
    aggregate = {}
    for arm in ("native",) + ARMS:
        subset = [record for record in all_records if record["arm"] == arm]
        require(len(subset) == len(packet["cases"]), "panel arm coverage changed")
        aggregate[arm] = {
            "tp50": sum(record["full_score"]["50"]["tp"] for record in subset),
            "fp50": sum(record["full_score"]["50"]["fp"] for record in subset),
            "strict_repeats": sum(record["full_score"]["strict_repeats"] for record in subset),
            "invalid_predictions": sum(record["full_score"]["invalid_predictions"] for record in subset),
            "parser_drops": sum(record["full_score"]["parser_drops"] for record in subset),
            "caps": sum(record["stop"] == "length" for record in subset),
            "eos": sum(record["stop"] == "eos" for record in subset),
        }
    result = {
        "status": "cold_panel_consumer_verified",
        "claim_boundary": packet["claim_boundary"],
        "scope": "Four pre-frozen exposed scenes only. Full and free suffix ledgers retained; supplied owners receive no autonomous credit; GT-unmatched predictions are not automatically hallucinations.",
        "packet_sha256": file_hash(packet_path),
        "launch_sha256": file_hash(out / "launch.json"),
        "stage_consumers": stage_consumers,
        "aggregate": aggregate,
        "records": all_records,
    }
    write(out / "consumer.json", result)
    return {"consumer": str(out / "consumer.json"), "sha256": file_hash(out / "consumer.json"), "cases": len(packet["cases"]), "records": len(all_records), "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "gate", "worker", "consume-stage", "launch", "consume-panel"))
    parser.add_argument("--packet", default=str(RAW_ROOT / "packet.json"))
    parser.add_argument("--out-dir")
    parser.add_argument("--gate-dir")
    parser.add_argument("--case-ids", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2))
        return
    require(args.out_dir, "fresh explicit --out-dir required")
    if args.command == "gate":
        print(json.dumps(gate(args.packet, args.out_dir), indent=2))
    elif args.command == "worker":
        run_stage(args.packet, args.out_dir, args.case_ids)
    elif args.command == "consume-stage":
        print(json.dumps(consume_stage(args.packet, args.out_dir), indent=2))
    elif args.command == "launch":
        require(args.gate_dir, "--gate-dir required")
        print(json.dumps(launch(args.packet, args.out_dir, args.gate_dir), indent=2))
    else:
        require(args.gate_dir, "--gate-dir required")
        print(json.dumps(consume_panel(args.packet, args.out_dir, args.gate_dir), indent=2))


if __name__ == "__main__":
    main()
