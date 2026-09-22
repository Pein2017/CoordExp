"""Offline native-template multimodal judging with auditable backend receipts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import statistics
import time
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_requests(path: Path, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("empty request population")
    seen = set()
    required = {"case_id", "image_path", "image_sha256", "system_prompt", "user_prompt"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"request missing fields: {required - row.keys()}")
        if row["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {row['case_id']}")
        seen.add(row["case_id"])
        if file_sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError(f"image hash mismatch: {row['case_id']}")
        if not all(isinstance(row[key], str) and row[key] for key in required):
            raise ValueError("request fields must be nonempty strings")
    if len({row["image_sha256"] for row in rows[:6]}) != len(rows[:6]):
        raise ValueError("cold/hot timing requires distinct first-six composite images")
    return rows


def render_prompt(processor: Any, row: dict) -> str:
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": [
            {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": row["user_prompt"]},
        ]},
    ]
    return processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def parse_answer(raw: str) -> tuple[dict | None, str | None]:
    """No repair, coercion, substring extraction, or retry of malformed answers."""
    try:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate answer key: {key}")
                result[key] = value
            return result
        answer = json.loads(raw.strip(), object_pairs_hook=unique_object)
        if not isinstance(answer, dict) or set(answer) != {"entity", "box"}:
            raise ValueError("answer must contain exactly entity and box")
        if any(not isinstance(value, str) or value not in {"yes", "no", "unknown"}
               for value in answer.values()):
            raise ValueError("answer values must be yes, no, or unknown")
        return answer, None
    except (ValueError, TypeError) as exc:
        return None, str(exc)


def timing_summary(batches: list[dict]) -> dict:
    hot = [batch for batch in batches if batch["phase"] == "hot_individual"]
    times = sorted(batch["wall_seconds"] for batch in hot)
    # Linear interpolation, identical to numpy.quantile(method='linear').
    position = (len(times) - 1) * .95
    low = int(position)
    p95 = (times[low] + (times[min(low + 1, len(times) - 1)] - times[low])
           * (position - low)) if times else None
    return {"hot_case_ids": [b["case_ids"][0] for b in hot],
            "hot_individual_median_seconds": statistics.median(times) if times else None,
            "hot_individual_p95_seconds": p95,
            "p95_method": "linear interpolation; descriptive only, at most five samples",
            "batches": batches}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(profile_path: Path, requests_path: Path, out_dir: Path,
        backend: str, limit: int | None = None) -> dict:
    from .profile import load_profile
    profile = load_profile(profile_path)
    rows = read_requests(requests_path, limit)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("this bounded run requires CUDA_VISIBLE_DEVICES=0")
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"status": "running", "backend": backend, "profile": profile,
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": file_sha256(profile_path),
                "requests_path": str(requests_path.resolve()),
                "requests_sha256": file_sha256(requests_path),
                "case_ids": [row["case_id"] for row in rows],
                "requested_count": len(rows), "generated_count": 0,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_sha256": file_sha256(__file__),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "vllm", "Pillow")},
                "parity_claim": "shared official checkpoint/template only; no numerical parity claim"}
    batches = []
    torch = None
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        torch.cuda.reset_peak_memory_stats()
        init_start = time.perf_counter()
        processor_kwargs = {key: profile[key] for key in ("min_pixels", "max_pixels")}
        processor = AutoProcessor.from_pretrained(profile["model_path"],
                                                 local_files_only=True, **processor_kwargs)
        prompts = [render_prompt(processor, row) for row in rows]
        images = []
        for row in rows:
            with Image.open(row["image_path"]) as source:
                images.append(source.convert("RGB"))
        if backend == "vllm":
            from vllm import LLM, SamplingParams
            settings = dict(model=profile["model_path"], dtype=profile["dtype"],
                            tensor_parallel_size=1, max_model_len=profile["max_model_len"],
                            max_num_seqs=profile["max_num_seqs"],
                            gpu_memory_utilization=profile["gpu_memory_utilization"],
                            enforce_eager=profile["enforce_eager"], seed=profile["seed"],
                            limit_mm_per_prompt={"image": 1},
                            mm_processor_kwargs=processor_kwargs,
                            enable_prefix_caching=False, mm_processor_cache_gb=0,
                            trust_remote_code=False)
            model = LLM(**settings)
            sampling = SamplingParams(temperature=0.0, max_tokens=profile["max_new_tokens"],
                                      seed=profile["seed"])
        elif backend == "transformers":
            from transformers import Qwen3VLForConditionalGeneration
            settings = dict(model=profile["model_path"], dtype=profile["dtype"],
                            device="cuda:0", attn_implementation="sdpa",
                            processor_kwargs=processor_kwargs, do_sample=False,
                            max_new_tokens=profile["max_new_tokens"])
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                profile["model_path"], dtype=getattr(torch, profile["dtype"]),
                attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
        else:
            raise ValueError(f"unsupported backend: {backend}")
        torch.cuda.synchronize()
        manifest["initialization_seconds"] = time.perf_counter() - init_start
        manifest["actual_settings"] = settings
        manifest["generation_settings"] = {"temperature": 0.0, "do_sample": False,
                                           "max_new_tokens": profile["max_new_tokens"],
                                           "seed": profile["seed"], "responses_per_request": 1}
        manifest["chat_template_sha256"] = canonical_sha256(processor.chat_template)
        manifest["gpu_name"] = torch.cuda.get_device_name(0)
        # Each primary request occurs once. No repeat-image or prefix-cache warm benchmark.
        groups = [[index] for index in range(min(6, len(rows)))]
        if len(rows) > 6:
            groups.append(list(range(6, len(rows))))
        with (out_dir / "responses.partial.jsonl").open("x") as output:
            initial_valid_answers = 0
            for batch_number, indices in enumerate(groups):
                phase = "cold_first_request" if batch_number == 0 else (
                    "hot_individual" if len(indices) == 1 and batch_number < 6 else "remaining_batch")
                torch.cuda.synchronize()
                batch_start = time.perf_counter()
                if backend == "vllm":
                    results = model.generate([
                        {"prompt": prompts[i], "multi_modal_data": {"image": images[i]}}
                        for i in indices], sampling, use_tqdm=False)
                    decoded = [(result.outputs[0].text, list(result.prompt_token_ids),
                                list(result.outputs[0].token_ids), result.outputs[0].finish_reason)
                               for result in results]
                else:
                    processor.tokenizer.padding_side = "left"
                    inputs = processor(text=[prompts[i] for i in indices],
                                       images=[images[i] for i in indices], padding=True,
                                       return_tensors="pt").to("cuda:0")
                    if inputs.input_ids.shape[1] + profile["max_new_tokens"] > profile["max_model_len"]:
                        raise ValueError("HF prompt plus generation exceeds profile max_model_len")
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False,
                                                   max_new_tokens=profile["max_new_tokens"])
                    suffix = generated[:, inputs.input_ids.shape[1]:]
                    decoded = []
                    eos = model.generation_config.eos_token_id
                    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
                    for j, tokens in enumerate(suffix.tolist()):
                        # Exclude only post-EOS padding, retaining the actual terminating token.
                        end = next((k + 1 for k, token in enumerate(tokens) if token in eos_ids), len(tokens))
                        tokens = tokens[:end]
                        prompt_ids = inputs.input_ids[j][inputs.attention_mask[j].bool()].tolist()
                        decoded.append((processor.decode(tokens, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False),
                                        prompt_ids, tokens,
                                        "stop" if tokens and tokens[-1] in eos_ids else "length"))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - batch_start
                if len(decoded) != len(indices):
                    raise RuntimeError("backend output count mismatch")
                receipt = {"phase": phase, "case_ids": [rows[i]["case_id"] for i in indices],
                           "wall_seconds": elapsed, "request_count": len(indices),
                           "requests_per_second": len(indices) / elapsed,
                           "amortized_seconds_per_request": elapsed / len(indices),
                           "generated_tokens": sum(len(item[2]) for item in decoded)}
                receipt["generated_tokens_per_second"] = receipt["generated_tokens"] / elapsed
                batches.append(receipt)
                for i, (raw, prompt_ids, token_ids, finish_reason) in zip(indices, decoded):
                    parsed, error = parse_answer(raw)
                    if batch_number < 6 and parsed is not None:
                        initial_valid_answers += 1
                    record = {"case_id": rows[i]["case_id"], "backend": backend,
                              "request_sha256": canonical_sha256(rows[i]),
                              "image_path": rows[i]["image_path"],
                              "image_sha256": rows[i]["image_sha256"],
                              "prompt_text": prompts[i], "prompt_token_ids": prompt_ids,
                              "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest(),
                              "prompt_tokens": len(prompt_ids), "raw_response": raw,
                              "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                              "finish_reason": finish_reason, "answer": parsed,
                              "parse_error": error, "batch_index": batch_number, "phase": phase,
                              "individual_wall_seconds": elapsed if len(indices) == 1 else None}
                    record["response_sha256"] = canonical_sha256(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest["generated_count"] += 1
                output.flush()
                os.fsync(output.fileno())
                print(json.dumps({"batch_complete": receipt,
                                  "generated_count": manifest["generated_count"]}), flush=True)
                if batch_number == 5 and not initial_valid_answers:
                    raise RuntimeError("all first-six answers failed parsing; refusing remaining batch")
        (out_dir / "responses.partial.jsonl").replace(out_dir / "responses.jsonl")
        manifest["responses_sha256"] = file_sha256(out_dir / "responses.jsonl")
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["total_wall_seconds"] = time.perf_counter() - started
        manifest["timing"] = timing_summary(batches)
        manifest["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch is not None and torch.cuda.is_initialized():
            manifest["driver_process_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            manifest["driver_process_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            manifest["gpu_memory_scope"] = "driver torch allocator only; vLLM worker memory not included"
        atomic_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.profile, args.requests, args.out_dir, args.backend, args.limit)


if __name__ == "__main__":
    main()
