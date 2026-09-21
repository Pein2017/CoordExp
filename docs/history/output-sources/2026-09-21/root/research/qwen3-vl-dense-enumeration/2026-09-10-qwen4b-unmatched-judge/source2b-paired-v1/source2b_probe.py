"""Bounded Source step2444 DoRA on/off instruction-following probe."""
from pathlib import Path
import hashlib
import json
import os
import time
from collections import Counter

import torch
from PIL import Image
from peft.tuners.tuners_utils import BaseTunerLayer

from probes.dora_owner_learning.runtime import DEFAULT_CONFIG, load_policy
from probes.unmatched_judge.infer import (
    atomic_json, canonical_sha256, file_sha256, parse_answer, read_requests, render_prompt,
)
from src.config.inference import load_research_infer_config

OUT = Path(__file__).resolve().parent
REQUESTS = OUT.parent / "source2b-paired-inputs/requests.jsonl"
MAX_NEW_TOKENS = 64


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def adapter_state(model, *, disabled):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            row = dict(name=name, disabled=bool(module.disable_adapters),
                       active_adapters=list(module.active_adapters),
                       merged_adapters=list(module.merged_adapters),
                       use_dora=dict(module.use_dora))
            if (row["disabled"] != disabled or row["active_adapters"] != ["default"]
                    or row["merged_adapters"] or row["use_dora"] != {"default": True}):
                raise RuntimeError(f"adapter switch invariant failed: {row}")
            layers.append(row)
    if len(layers) != 196:
        raise RuntimeError(f"expected 196 DoRA target modules, found {len(layers)}")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("all model and selected embedding parameters must remain frozen")
    delta_in = model.get_input_embeddings().shared_embed_delta
    delta_out = model.get_output_embeddings().shared_embed_delta
    if delta_in is not delta_out:
        raise RuntimeError("selected input/output embedding delta must remain tied")
    selected = {name: tensor_hash(p) for name, p in model.named_parameters()
                if "lora_" in name or "shared_embed_delta" in name}
    return dict(layer_count=len(layers), layers=layers, tensor_hashes=selected,
                tensor_hashes_sha256=canonical_sha256(selected),
                selected_embedding_sha256=tensor_hash(delta_in),
                selected_embedding_shape=list(delta_in.shape),
                selected_embedding_tied=True, trainable_parameters=0,
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                model_object_id=id(model))


def freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def output_kind(raw, answer):
    if answer is not None:
        return "valid_two_field_json"
    if any(marker in raw for marker in ("<|object_ref_start|>", "<|box_start|>", "<|coord_")):
        return "detection_tokens_instead_of_requested_json"
    try:
        json.loads(raw)
    except ValueError:
        return "non_json_text"
    return "json_schema_failure"


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this task owns only GPU 0")
    if (OUT / "manifest.json").exists() or (OUT / "dora-on").exists():
        raise RuntimeError("refusing to overwrite an existing paired run")
    started = time.perf_counter()
    manifest = dict(status="running", requested_count=24, generated_count=0,
                    contrast="conditional DoRA increment, selected embedding delta held fixed",
                    not_claimed="pristine official base versus all finetuning",
                    execution_order=["dora-on", "dora-off"],
                    source_sha256=file_sha256(__file__),
                    requests_sha256=file_sha256(REQUESTS),
                    source_config_path=str(DEFAULT_CONFIG),
                    source_config_sha256=file_sha256(DEFAULT_CONFIG),
                    CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
                    generation=dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    repetition_penalty=1.0, use_cache=True), arms={})
    model = None
    try:
        rows = read_requests(REQUESTS)
        if len(rows) != 12:
            raise RuntimeError("frozen population requires exactly 12 requests")
        resolved = load_research_infer_config(DEFAULT_CONFIG)
        init = time.perf_counter()
        qwen, descriptor = load_policy(resolved.config, device="cuda:0")
        model = qwen.model
        freeze(model)
        torch.cuda.synchronize()
        manifest["initialization_seconds"] = time.perf_counter() - init
        manifest["loaded_policy"] = descriptor
        manifest["qwen_components"] = qwen.to_artifact_dict()
        manifest["source_config_fingerprint"] = resolved.fingerprint
        manifest["gpu_name"] = torch.cuda.get_device_name(0)
        manifest["torch_num_threads"] = torch.get_num_threads()
        initial_state = adapter_state(model, disabled=False)
        atomic_json(OUT / "state-initial-enabled.json", initial_state)
        parameter_versions = {name: p._version for name, p in model.named_parameters()}
        preprocessing = time.perf_counter()
        cache = []
        for row in rows:
            prompt = render_prompt(qwen.processor, row)
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                inputs = qwen.processor(text=[prompt], images=[image], do_resize=False,
                                        return_tensors="pt", padding=False)
            hashes = {name: tensor_hash(value) for name, value in inputs.items()}
            cache.append((row, prompt, inputs, hashes))
        manifest["preprocessing_once_seconds"] = time.perf_counter() - preprocessing
        manifest["processor_settings"] = {"do_resize": False, "native_chat_template": True,
                                          "same_cpu_input_tensor_objects_reused_for_both_arms": True}
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        for arm, disabled in (("dora-on", False), ("dora-off", True)):
            if disabled:
                model.disable_adapters()
                freeze(model)
            state = adapter_state(model, disabled=disabled)
            if state["tensor_hashes"] != initial_state["tensor_hashes"]:
                raise RuntimeError("adapter or selected embedding weights changed across toggle")
            arm_dir = OUT / arm
            arm_dir.mkdir(exist_ok=False)
            atomic_json(arm_dir / "state-before.json", state)
            results = []
            torch.cuda.reset_peak_memory_stats()
            arm_started = time.perf_counter()
            with (arm_dir / "responses.partial.jsonl").open("x") as handle:
                for row, prompt, cpu_inputs, hashes in cache:
                    if {name: tensor_hash(value) for name, value in cpu_inputs.items()} != hashes:
                        raise RuntimeError("cached input tensor mutation")
                    torch.cuda.synchronize()
                    request_started = time.perf_counter()
                    inputs = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                                   do_sample=False, repetition_penalty=1.0,
                                                   use_cache=True)
                    tokens = generated[0, inputs["input_ids"].shape[1]:].tolist()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - request_started
                    ended = bool(tokens and tokens[-1] in eos_ids)
                    content_tokens = tokens[:-1] if ended else tokens
                    raw = qwen.tokenizer.decode(content_tokens, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                    answer, error = parse_answer(raw)
                    record = dict(case_id=row["case_id"], arm=arm, answer=answer,
                                  parse_error=error, output_kind=output_kind(raw, answer),
                                  raw_response=raw,
                                  raw_response_including_terminal_specials=qwen.tokenizer.decode(
                                      tokens, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False),
                                  request_sha256=canonical_sha256(row),
                                  image_sha256=row["image_sha256"], image_path=row["image_path"],
                                  prompt_text=prompt,
                                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                  input_tensor_sha256=hashes,
                                  prompt_token_ids=cpu_inputs["input_ids"][0].tolist(),
                                  prompt_tokens=cpu_inputs["input_ids"].shape[1],
                                  image_grid_thw=cpu_inputs["image_grid_thw"].tolist(),
                                  generated_token_ids=tokens, generated_tokens=len(tokens),
                                  finish_reason="stop" if ended else "length",
                                  inference_wall_seconds=elapsed)
                    record["response_sha256"] = canonical_sha256(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    results.append(record)
                    manifest["generated_count"] += 1
                    print(json.dumps({"arm": arm, "case_id": row["case_id"],
                                      "output_kind": record["output_kind"], "answer": answer,
                                      "seconds": elapsed}), flush=True)
                    del inputs, generated
            (arm_dir / "responses.partial.jsonl").replace(arm_dir / "responses.jsonl")
            manifest["arms"][arm] = dict(count=len(results),
                output_kind_counts=dict(Counter(r["output_kind"] for r in results)),
                answer_counts=dict(Counter(str(r["answer"]) for r in results)),
                wall_seconds=time.perf_counter() - arm_started,
                inference_wall_seconds_sum=sum(r["inference_wall_seconds"] for r in results),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                responses_sha256=file_sha256(arm_dir / "responses.jsonl"))
        model.enable_adapters()
        freeze(model)
        restored = adapter_state(model, disabled=False)
        if restored["tensor_hashes"] != initial_state["tensor_hashes"]:
            raise RuntimeError("weights changed after restoring adapters")
        if {name: p._version for name, p in model.named_parameters()} != parameter_versions:
            raise RuntimeError("parameter version changed during paired inference")
        atomic_json(OUT / "state-restored-enabled.json", restored)
        manifest["same_model_object_both_arms"] = True
        manifest["all_parameter_versions_unchanged"] = True
        manifest["adapter_and_selected_embedding_hashes_unchanged"] = True
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model is not None:
            model.enable_adapters()
            freeze(model)
        manifest["total_wall_seconds"] = time.perf_counter() - started
        atomic_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
