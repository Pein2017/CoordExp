#!/usr/bin/env python3
"""Qualify one real Qwen3-VL image through offline vLLM.

The probe is intentionally executable evidence rather than a unit-test double.
It validates the exact no-resize prompt ids, generated processed likelihoods,
raw teacher-forced replay likelihoods, stop-token retention, CUDA isolation,
and engine cleanup required by the vLLM OpenSpec change.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = Path(
    "configs/coordexp_swift/infer/wave7_real_base_single_smoke.yaml"
)
DEFAULT_RECEIPT = Path(
    "src/inference/qualification_receipts/vllm-0.14.1-qualification.json"
)
STATIC_SOURCE_FILES = {
    "qwen3_vl": "model_executor/models/qwen3_vl.py",
    "model_config": "config/model.py",
    "logprobs": "logprobs.py",
    "peft_helper": "lora/peft_helper.py",
    "llm_entrypoint": "entrypoints/llm.py",
    "sampling_params": "sampling_params.py",
    "prompt_inputs": "inputs/data.py",
    "multimodal_processing": "multimodal/processing.py",
    "transformers_processor": "transformers_utils/processor.py",
    "v1_llm_engine": "v1/engine/llm_engine.py",
    "v1_engine_core": "v1/engine/core.py",
    "distributed_parallel_state": "distributed/parallel_state.py",
}
REQUIRED_LOADED_SOURCE_PATHS = {
    "vllm/model_executor/model_loader/default_loader.py",
    "vllm/model_executor/models/qwen3_vl.py",
    "vllm/multimodal/processing.py",
    "vllm/v1/executor/uniproc_executor.py",
    "vllm/v1/sample/ops/topk_topp_sampler.py",
    "vllm/v1/sample/sampler.py",
    "vllm/v1/worker/gpu_model_runner.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help=(
            "Explicit KV-cache allocation. This avoids vLLM's free-memory "
            "profiling race on shared GPUs while retaining a real profile run."
        ),
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument(
        "--engine-process-mode",
        choices=("uniprocess", "multiprocess_spawn"),
        default="uniprocess",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not args.worker:
        _run_parent(args)
        return
    _run_worker(args)


def _run_parent(args: argparse.Namespace) -> None:
    import psutil

    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{receipt_path.stem}-worker-",
        suffix=".json",
        dir=receipt_path.parent,
        delete=False,
    ) as handle:
        worker_receipt_path = Path(handle.name)
    worker_receipt_path.unlink()

    parent = psutil.Process()
    children_before = _child_processes(parent)
    gpu_memory_before = _visible_gpu_memory_mib()
    command = _worker_command(args, receipt=worker_receipt_path)
    completed = subprocess.run(command, check=False)
    time.sleep(1.0)
    children_after = _child_processes(parent)
    gpu_memory_after = _visible_gpu_memory_mib()

    if worker_receipt_path.exists():
        receipt = json.loads(worker_receipt_path.read_text(encoding="utf-8"))
        worker_receipt_path.unlink()
    else:
        receipt = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "candidate_version": "0.14.1",
            "status": "failed",
            "failure": {
                "type": "MissingWorkerReceipt",
                "message": "qualification worker exited without a receipt",
            },
        }

    worker_pid = receipt.get("process", {}).get("pid")
    memory_released = gpu_memory_after <= gpu_memory_before + 64
    post_exit = {
        "launcher_pid": os.getpid(),
        "launcher_argv": list(sys.argv),
        "worker_command": command,
        "worker_returncode": int(completed.returncode),
        "worker_pid": worker_pid,
        "worker_pid_alive_after_exit": (
            bool(psutil.pid_exists(int(worker_pid))) if worker_pid is not None else None
        ),
        "children_before": children_before,
        "children_after": children_after,
        "gpu_memory_before_mib": gpu_memory_before,
        "gpu_memory_after_mib": gpu_memory_after,
        "gpu_memory_returned_to_baseline": memory_released,
        "gpu_memory_tolerance_mib": 64,
    }
    receipt["post_worker_exit"] = post_exit
    passed = (
        receipt.get("status") == "passed"
        and completed.returncode == 0
        and not children_after
        and post_exit["worker_pid_alive_after_exit"] is False
        and memory_released
    )
    if not passed:
        receipt["status"] = "failed"
        receipt.setdefault(
            "failure",
            {
                "type": "PostWorkerExitFailure",
                "message": "qualification worker did not satisfy post-exit cleanup",
            },
        )
    _write_receipt(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}, sort_keys=True))
    if not passed:
        raise SystemExit(completed.returncode or 1)


def _run_worker(args: argparse.Namespace) -> None:
    process_mode = _configure_process_mode(args.engine_process_mode)

    import psutil
    import torch

    parent = psutil.Process()
    children_before = _child_processes(parent)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "probe": {
            "path": str(Path(__file__).resolve()),
            "repo_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_file(Path(__file__).resolve()),
            "argv": list(sys.argv),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_version": "0.14.1",
        "status": "running",
        "process": {
            **process_mode,
            "pid": os.getpid(),
            "children_before": children_before,
        },
        "cuda": _cuda_receipt(torch),
    }
    llm: Any | None = None
    failure: BaseException | None = None
    started = time.monotonic()
    try:
        _validate_cuda(torch)
        resolved, raw_example, components, prompt_record = _prepare_fixture(args)

        llm, engine_receipt = _open_engine(
            args=args,
            resolved=resolved,
        )
        receipt["engine"] = engine_receipt
        receipt["process"]["children_after_engine_open"] = _child_processes(parent)
        receipt.update(
            _identity_receipt(
                args=args,
                resolved=resolved,
                raw_example=raw_example,
                components=components,
                prompt_record=prompt_record,
                llm=llm,
            )
        )

        generated = _run_generation(
            llm=llm,
            args=args,
            prompt_record=prompt_record,
            raw_example=raw_example,
            components=components,
        )
        replay = _run_raw_replay(
            llm=llm,
            prompt_record=prompt_record,
            generated_token_ids=generated["generated_token_ids"],
            raw_example=raw_example,
            components=components,
        )
        if generated["executed_media"] != replay["executed_media"]:
            raise RuntimeError(
                "generation and raw replay did not execute identical image bytes"
            )
        likelihood_alignment = _align_likelihoods(
            generated_token_ids=generated["generated_token_ids"],
            policy_logprobs=generated["policy_logprobs"],
            raw_model_logprobs=replay["raw_model_logprobs"],
        )
        receipt["generation"] = generated
        receipt["raw_replay"] = replay
        receipt["likelihood_alignment"] = likelihood_alignment
        receipt["loaded_runtime_sources"] = _loaded_runtime_source_manifest()
        receipt["status"] = "passed"
    except BaseException as exc:  # Preserve the concrete qualification failure.
        failure = exc
        receipt["status"] = "failed"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        cleanup = _close_engine(llm, torch=torch, parent=parent)
        llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cuda_after_cleanup = _cuda_receipt(torch)
        cleanup["cuda_after_outer_release"] = cuda_after_cleanup
        receipt["cleanup"] = cleanup
        receipt["duration_seconds"] = time.monotonic() - started
        receipt["cuda_after_cleanup"] = cuda_after_cleanup
        if cleanup["owned_children_after_cleanup"]:
            receipt["status"] = "failed"
            cleanup_error = RuntimeError(
                "vLLM qualification left live child processes after cleanup"
            )
            if failure is None:
                failure = cleanup_error
                receipt["failure"] = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
        _write_receipt(args.receipt, receipt)

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.receipt.resolve()),
            },
            sort_keys=True,
        )
    )
    if failure is not None:
        raise failure


def _worker_command(args: argparse.Namespace, *, receipt: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(args.config),
        "--receipt",
        str(receipt),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--engine-process-mode",
        args.engine_process_mode,
    ]


def _visible_gpu_memory_mib() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError("qualification launcher requires CUDA_VISIBLE_DEVICES")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(tokens) != 1:
        raise RuntimeError(
            f"qualification launcher requires one visible GPU token, got {tokens}"
        )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            tokens[0],
        ],
        text=True,
    ).strip()
    return int(output.splitlines()[0].strip())


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max-model-len must exceed --max-new-tokens")
    if args.kv_cache_memory_bytes <= 0:
        raise ValueError("--kv-cache-memory-bytes must be positive")
    if args.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be positive")


def _configure_process_mode(mode: str) -> dict[str, Any]:
    if mode == "uniprocess":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    else:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return {
        "engine_process_mode": mode,
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ[
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ],
        "VLLM_WORKER_MULTIPROC_METHOD": os.environ[
            "VLLM_WORKER_MULTIPROC_METHOD"
        ],
    }


def _prepare_fixture(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from src.config.inference import load_infer_config
    from src.config.models import ProcessorConfig
    from src.data import load_raw_examples
    from src.inference.image_plan import plan_image_batch
    from src.inference.prompt import build_prompt_record
    from src.qwen.runtime_loading import (
        QwenLoadOptions,
        load_qwen_components_from_options,
    )

    resolved = load_infer_config(args.config)
    raw_examples = load_raw_examples(resolved.config.data.input_jsonl)
    if len(raw_examples) != 1:
        raise RuntimeError(
            f"qualification fixture must contain exactly one row, got {len(raw_examples)}"
        )
    raw_example = raw_examples[0]
    if resolved.config.backend.type == "hf":
        attention = resolved.config.backend.hf.attn_implementation
        patch_policy = resolved.config.backend.hf.patch_embed_linearization
    else:
        attention = "eager"
        patch_policy = "disabled"
    components = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=resolved.config.model.base_model,
            dtype=resolved.config.model.dtype,
            attn_implementation=attention,
            patch_embed_linearization=patch_policy,
            load_model=False,
        )
    )
    image_row = plan_image_batch(
        [raw_example],
        components=components,
        processor_config=ProcessorConfig(
            do_resize=False,
            max_raw_pixels=1_000_000_000,
            max_merged_visual_tokens=1_000_000,
        ),
        row_indices=[0],
    ).rows[0]
    prompt_record = build_prompt_record(
        raw_example,
        resolved.config.template,
        processor=components.processor,
        row_index=0,
        merged_visual_tokens=image_row.merged_visual_tokens,
    )
    return resolved, raw_example, components, prompt_record


def _open_engine(*, args: argparse.Namespace, resolved: Any) -> tuple[Any, dict[str, Any]]:
    import vllm
    from vllm import LLM

    dtype = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[resolved.config.model.dtype]
    kwargs = {
        "model": resolved.config.model.base_model,
        "tokenizer": resolved.config.model.base_model,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": dtype,
        "seed": 0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "disable_custom_all_reduce": True,
        "disable_log_stats": True,
        "generation_config": "vllm",
        "logprobs_mode": "processed_logprobs",
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"do_resize": False},
    }
    llm = LLM(**kwargs)
    model_config = llm.llm_engine.model_config
    return llm, {
        "vllm_version": vllm.__version__,
        "engine_class": type(llm.llm_engine).__name__,
        "entrypoint_class": type(llm).__name__,
        "effective_kwargs": kwargs,
        "qualification_argument_policy": {
            "semantic_invariants": {
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "dtype": dtype,
                "logprobs_mode": "processed_logprobs",
                "generation_config": "vllm",
                "limit_mm_per_prompt": {"image": 1, "video": 0},
                "mm_processor_kwargs": {"do_resize": False},
            },
            "qualified_exact_values": {
                "max_model_len": args.max_model_len,
                "max_num_seqs": 1,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
            "harness_only_values": {
                "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                "reason": "avoid shared-GPU free-memory profiling races",
            },
            "requires_additional_executed_probe": [
                "max_num_seqs",
                "max_model_len",
                "gpu_memory_utilization",
                "production_automatic_kv_cache_sizing",
            ],
        },
        "model_config": {
            "model": str(model_config.model),
            "dtype": str(model_config.dtype),
            "max_model_len": int(model_config.max_model_len),
            "logprobs_mode": str(model_config.logprobs_mode),
        },
    }


def _run_generation(
    *,
    llm: Any,
    args: argparse.Namespace,
    prompt_record: Any,
    raw_example: Any,
    components: Any,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TextPrompt

    image, executed_media = _load_hashed_rgb_image(raw_example)
    im_end_id = int(components.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    prompt = TextPrompt(
        prompt=prompt_record.chat_text,
        multi_modal_data={"image": image},
        mm_processor_kwargs={"do_resize": False},
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        logprobs=0,
        stop_token_ids=[im_end_id],
        ignore_eos=False,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("qualification generation must return one request and one output")
    request_output = outputs[0]
    output = request_output.outputs[0]
    prompt_ids = [int(item) for item in request_output.prompt_token_ids or ()]
    expected_prompt_ids = [int(item) for item in prompt_record.prompt_token_ids]
    if prompt_ids != expected_prompt_ids:
        raise RuntimeError(
            "vLLM prompt ids differ from training-aligned no-resize prompt ids: "
            f"observed={len(prompt_ids)} expected={len(expected_prompt_ids)}"
        )
    generated_ids = [int(item) for item in output.token_ids]
    if not generated_ids:
        raise RuntimeError("vLLM qualification generated no tokens")
    if generated_ids[-1] != im_end_id:
        raise RuntimeError(
            "vLLM qualification did not retain terminal <|im_end|>: "
            f"last={generated_ids[-1]} finish_reason={output.finish_reason!r} "
            f"stop_reason={output.stop_reason!r}"
        )
    if output.finish_reason != "stop":
        raise RuntimeError(
            f"vLLM qualification expected stop finish_reason, got {output.finish_reason!r}"
        )
    if output.logprobs is None or len(output.logprobs) != len(generated_ids):
        raise RuntimeError("vLLM generated-token logprob length mismatch")
    policy_logprobs = _chosen_logprobs(
        token_ids=generated_ids,
        positions=output.logprobs,
        channel="policy",
    )
    decoded = components.tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    expected_native_text = components.tokenizer.decode(
        generated_ids[:-1],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if output.text != expected_native_text:
        raise RuntimeError(
            "vLLM native text differs from tokenizer decode before the terminal "
            f"stop token: vllm={output.text!r} tokenizer={expected_native_text!r}"
        )
    image_pad_id = int(components.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    expected_visual_tokens = expected_prompt_ids.count(image_pad_id)
    placeholder_ranges = _image_pad_ranges(prompt_ids, image_pad_id=image_pad_id)
    observed_visual_tokens = sum(item["count"] for item in placeholder_ranges)
    if expected_visual_tokens <= 1:
        raise RuntimeError("no-resize prompt did not contain expanded image placeholders")
    if observed_visual_tokens != expected_visual_tokens:
        raise RuntimeError(
            "returned prompt image-placeholder count differs from expected no-resize "
            f"expansion: observed={observed_visual_tokens} "
            f"expected={expected_visual_tokens}"
        )
    return {
        "request_id": str(request_output.request_id),
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "image_pad_token_id": image_pad_id,
        "expanded_image_pad_count": observed_visual_tokens,
        "validated_multi_modal_placeholder_ranges": placeholder_ranges,
        "backend_reported_multi_modal_placeholders": _jsonable(
            request_output.multi_modal_placeholders
        ),
        "executed_media": executed_media,
        "generated_token_ids": generated_ids,
        "generated_token_texts": [
            components.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in generated_ids
        ],
        "native_generated_text": output.text,
        "raw_generated_text": decoded,
        "native_text_omits_terminal_im_end": True,
        "raw_text_reconstructed_from_authoritative_token_ids": True,
        "finish_reason": output.finish_reason,
        "native_stop_reason": output.stop_reason,
        "stop_reason": "im_end",
        "im_end_token_id": im_end_id,
        "im_end_retained": True,
        "policy_logprobs": policy_logprobs,
        "cumulative_policy_logprob": float(output.cumulative_logprob),
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_new_tokens,
            "logprobs": 0,
        },
    }


def _run_raw_replay(
    *,
    llm: Any,
    prompt_record: Any,
    generated_token_ids: list[int],
    raw_example: Any,
    components: Any,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    image, executed_media = _load_hashed_rgb_image(raw_example)
    unexpanded_prompt_ids = [
        int(item)
        for item in components.tokenizer(
            prompt_record.chat_text,
            add_special_tokens=False,
        )["input_ids"]
    ]
    replay_prompt = TokensPrompt(
        prompt_token_ids=unexpanded_prompt_ids + list(generated_token_ids),
        multi_modal_data={"image": image},
        mm_processor_kwargs={"do_resize": False},
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_tokens=1,
        prompt_logprobs=0,
        logprobs=None,
        ignore_eos=True,
        detokenize=False,
        skip_special_tokens=False,
    )
    outputs = llm.generate([replay_prompt], sampling, use_tqdm=False)
    if len(outputs) != 1:
        raise RuntimeError("raw replay must return exactly one request")
    replay_output = outputs[0]
    expected_ids = list(prompt_record.prompt_token_ids) + list(generated_token_ids)
    observed_ids = [int(item) for item in replay_output.prompt_token_ids or ()]
    if observed_ids != expected_ids:
        raise RuntimeError(
            "raw replay prompt ids differ from prompt plus generated ids: "
            f"observed={len(observed_ids)} expected={len(expected_ids)}"
        )
    prompt_logprobs = replay_output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_ids):
        raise RuntimeError(
            "raw replay prompt-logprob length mismatch: "
            f"observed={None if prompt_logprobs is None else len(prompt_logprobs)} "
            f"expected={len(expected_ids)}"
        )
    image_pad_id = int(components.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    placeholder_ranges = _image_pad_ranges(observed_ids, image_pad_id=image_pad_id)
    expected_placeholder_ranges = _image_pad_ranges(
        list(prompt_record.prompt_token_ids),
        image_pad_id=image_pad_id,
    )
    if placeholder_ranges != expected_placeholder_ranges:
        raise RuntimeError(
            "raw replay image-placeholder ranges differ from generation prefix"
        )
    start = len(prompt_record.prompt_token_ids)
    generated_positions = prompt_logprobs[start : start + len(generated_token_ids)]
    raw_logprobs = _chosen_logprobs(
        token_ids=generated_token_ids,
        positions=generated_positions,
        channel="raw_model",
    )
    replay_generated_ids = [
        int(item) for item in replay_output.outputs[0].token_ids
    ]
    return {
        "unexpanded_prompt_token_ids": unexpanded_prompt_ids,
        "expanded_prompt_plus_generated_token_ids": observed_ids,
        "generated_absolute_start": start,
        "generated_absolute_end": start + len(generated_token_ids),
        "raw_model_logprobs": raw_logprobs,
        "validated_multi_modal_placeholder_ranges": placeholder_ranges,
        "backend_reported_multi_modal_placeholders": _jsonable(
            replay_output.multi_modal_placeholders
        ),
        "executed_media": executed_media,
        "replay_only_generated_token_ids": replay_generated_ids,
        "replay_only_generated_tokens_discarded": True,
        "prompt_logprobs": 0,
    }


def _chosen_logprobs(
    *,
    token_ids: list[int],
    positions: Any,
    channel: str,
) -> list[float]:
    values: list[float] = []
    if len(positions) != len(token_ids):
        raise RuntimeError(
            f"{channel} likelihood position count differs from token count"
        )
    for index, (token_id, candidates) in enumerate(zip(token_ids, positions, strict=True)):
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                f"{channel} likelihood missing chosen token at step {index}: {token_id}"
            )
        value = float(candidates[token_id].logprob)
        if not math.isfinite(value) or value > 1e-6:
            raise RuntimeError(
                f"{channel} likelihood is invalid at step {index}: {value}"
            )
        values.append(value)
    return values


def _align_likelihoods(
    *,
    generated_token_ids: list[int],
    policy_logprobs: list[float],
    raw_model_logprobs: list[float],
) -> dict[str, Any]:
    if not (
        len(generated_token_ids)
        == len(policy_logprobs)
        == len(raw_model_logprobs)
    ):
        raise RuntimeError("policy/raw likelihood channels are not token aligned")
    absolute_deltas = [
        abs(policy - raw)
        for policy, raw in zip(policy_logprobs, raw_model_logprobs, strict=True)
    ]
    return {
        "token_count": len(generated_token_ids),
        "token_ids_aligned": True,
        "finite_non_positive": True,
        "different_value_count_at_1e_6": sum(delta > 1e-6 for delta in absolute_deltas),
        "max_absolute_channel_delta": max(absolute_deltas, default=0.0),
    }


def _identity_receipt(
    *,
    args: argparse.Namespace,
    resolved: Any,
    raw_example: Any,
    components: Any,
    prompt_record: Any,
    llm: Any,
) -> dict[str, Any]:
    import vllm

    from src.inference.model_assets import build_model_snapshot_manifest

    vllm_root = Path(vllm.__file__).resolve().parent
    source_files = {
        name: _installed_file_identity(
            vllm_root / relative,
            package="vllm",
            package_root=vllm_root,
        )
        for name, relative in STATIC_SOURCE_FILES.items()
    }
    source_files.update(_runtime_source_identities(llm=llm, components=components))
    base_root = Path(resolved.config.model.base_model).resolve()
    model_manifest = build_model_snapshot_manifest(base_root)
    wrapper_tokens = (
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|im_end|>",
    )
    wrapper_ids = {
        token: int(components.tokenizer.convert_tokens_to_ids(token))
        for token in wrapper_tokens
    }
    if len(set(wrapper_ids.values())) != len(wrapper_ids):
        raise RuntimeError("qualification special wrapper ids are not unique")
    for token, token_id in wrapper_ids.items():
        if components.tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise RuntimeError(f"qualification wrapper is not one token: {token}")
    input_prompt_token_ids = [
        int(item)
        for item in components.tokenizer(
            prompt_record.chat_text,
            add_special_tokens=False,
        )["input_ids"]
    ]
    image_pad_id = int(components.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    input_image_pad_count = input_prompt_token_ids.count(image_pad_id)
    executed_image_pad_count = prompt_record.prompt_token_ids.count(image_pad_id)
    if input_image_pad_count != 1:
        raise RuntimeError(
            "qualification input prompt must contain exactly one image placeholder, "
            f"got {input_image_pad_count}"
        )
    if executed_image_pad_count <= input_image_pad_count:
        raise RuntimeError("qualification expected prompt did not expand image placeholders")
    return {
        "dependencies": {
            name: _package_version(name)
            for name in ("vllm", "transformers", "peft", "torch", "qwen-vl-utils")
        },
        "installed_sources": source_files,
        "fixture": {
            "config": _file_identity(args.config.resolve()),
            "input_jsonl": _file_identity(Path(resolved.config.data.input_jsonl)),
            "row_id": raw_example.example_id,
            "image": _file_identity(Path(raw_example.image.path)),
            "image_width": raw_example.image.width,
            "image_height": raw_example.image.height,
            "prompt_text_sha256": _sha256_bytes(
                prompt_record.chat_text.encode("utf-8")
            ),
            "input_prompt_token_ids": input_prompt_token_ids,
            "input_prompt_token_count": len(input_prompt_token_ids),
            "input_image_pad_count": input_image_pad_count,
            "expected_executed_prompt_token_ids": list(
                prompt_record.prompt_token_ids
            ),
            "expected_prompt_token_count": len(prompt_record.prompt_token_ids),
            "expected_executed_image_pad_count": executed_image_pad_count,
        },
        "model": {
            "base_model": str(base_root),
            "snapshot_manifest": model_manifest,
            "components": components.to_artifact_dict(),
        },
        "qualification_scope": {
            "kind": "base-family-runtime-qualification",
            "source_base_snapshot_fingerprint": model_manifest["fingerprint"],
            "model_architecture": components.model_identity.to_artifact_dict(),
            "tokenizer_identity": components.token_identity.to_artifact_dict(),
            "processor_identity": components.processor_identity.to_artifact_dict(),
            "execution_model_acceptance": {
                "base_only": "exact snapshot fingerprint",
                "composed": (
                    "matching source-base, architecture, tokenizer, and processor "
                    "identity plus a passed execution-model composition-fidelity receipt"
                ),
            },
        },
        "special_tokens": {
            "single_token_verified": True,
            "ids": wrapper_ids,
        },
    }


def _runtime_source_identities(*, llm: Any, components: Any) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.inputs import TextPrompt, TokensPrompt

    engine = llm.llm_engine
    engine_core = engine.engine_core
    objects = {
        "runtime_llm_class": LLM,
        "runtime_sampling_params": SamplingParams,
        "runtime_text_prompt": TextPrompt,
        "runtime_tokens_prompt": TokensPrompt,
        "runtime_engine_class": type(engine),
        "runtime_engine_core_class": type(engine_core),
        "runtime_engine_core_shutdown": type(engine_core).shutdown,
        "runtime_destroy_model_parallel": destroy_model_parallel,
        "runtime_destroy_distributed_environment": destroy_distributed_environment,
        "runtime_processor_class": type(components.processor),
        "runtime_tokenizer_class": type(components.tokenizer),
        "runtime_image_processor_class": type(components.processor.image_processor),
    }
    import peft
    import qwen_vl_utils
    import transformers
    import vllm

    package_roots = {
        "peft": Path(peft.__file__).resolve().parent,
        "qwen_vl_utils": Path(qwen_vl_utils.__file__).resolve().parent,
        "transformers": Path(transformers.__file__).resolve().parent,
        "vllm": Path(vllm.__file__).resolve().parent,
    }
    identities: dict[str, Any] = {}
    for name, value in objects.items():
        source = inspect.getsourcefile(value) or inspect.getfile(value)
        if source is None:
            raise RuntimeError(f"cannot locate installed source for {name}")
        source_path = Path(source).resolve()
        owner = _source_package_owner(source_path, packages=package_roots)
        if owner is None:
            raise RuntimeError(f"installed source is outside qualified packages: {source_path}")
        package_name, package_root = owner
        identities[name] = _installed_file_identity(
            source_path,
            package=package_name,
            package_root=package_root,
        )
    return identities


def _loaded_runtime_source_manifest() -> dict[str, Any]:
    """Hash every loaded source module from the qualified package boundary."""

    import peft
    import qwen_vl_utils
    import transformers
    import vllm

    packages = {
        "peft": Path(peft.__file__).resolve().parent,
        "qwen_vl_utils": Path(qwen_vl_utils.__file__).resolve().parent,
        "transformers": Path(transformers.__file__).resolve().parent,
        "vllm": Path(vllm.__file__).resolve().parent,
    }
    by_path: dict[Path, dict[str, Any]] = {}
    for module_name, module in sorted(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        path = Path(module_file).resolve()
        if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        if not path.is_file():
            continue
        owner = _source_package_owner(path, packages=packages)
        if owner is None:
            continue
        package_name, package_root = owner
        row = by_path.setdefault(
            path,
            {
                "package": package_name,
                "relative_path": (
                    Path(package_name) / path.relative_to(package_root)
                ).as_posix(),
                **_file_identity(path),
                "modules": [],
            },
        )
        row["modules"].append(module_name)

    files = []
    for row in sorted(by_path.values(), key=lambda item: item["relative_path"]):
        row["modules"] = sorted(set(row["modules"]))
        files.append(row)
    observed_paths = {row["relative_path"] for row in files}
    missing_required = sorted(REQUIRED_LOADED_SOURCE_PATHS - observed_paths)
    if missing_required:
        raise RuntimeError(
            "qualification did not execute required vLLM source owners: "
            f"{missing_required}"
        )
    fingerprint_payload = [
        {
            "package": row["package"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "modules": row["modules"],
        }
        for row in files
    ]
    return {
        "policy": "all-loaded-vllm-transformers-peft-qwen-vl-utils-modules-v1",
        "package_roots": {
            name: str(root) for name, root in sorted(packages.items())
        },
        "file_count": len(files),
        "files": files,
        "required_paths": sorted(REQUIRED_LOADED_SOURCE_PATHS),
        "required_paths_present": True,
        "fingerprint": _sha256_json(fingerprint_payload),
    }


def _source_package_owner(
    path: Path,
    *,
    packages: dict[str, Path],
) -> tuple[str, Path] | None:
    for name, root in packages.items():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return name, root
    return None


def _load_hashed_rgb_image(raw_example: Any) -> tuple[Any, dict[str, Any]]:
    from PIL import Image

    image_path = Path(raw_example.image.path).resolve()
    source_bytes = image_path.read_bytes()
    with Image.open(io.BytesIO(source_bytes)) as source:
        image = source.convert("RGB").copy()
    expected_size = (raw_example.image.width, raw_example.image.height)
    if image.size != expected_size:
        raise RuntimeError(
            "qualification image dimensions differ from the validated raw example: "
            f"decoded={image.size} declared={expected_size}"
        )
    pixel_payload = (
        f"RGB:{image.width}x{image.height}:".encode("ascii") + image.tobytes()
    )
    return image, {
        "path": str(image_path),
        "source_size_bytes": len(source_bytes),
        "source_sha256": _sha256_bytes(source_bytes),
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "rgb_pixel_sha256": _sha256_bytes(pixel_payload),
    }


def _image_pad_ranges(
    token_ids: list[int],
    *,
    image_pad_id: int,
) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] != image_pad_id:
            cursor += 1
            continue
        start = cursor
        while cursor < len(token_ids) and token_ids[cursor] == image_pad_id:
            cursor += 1
        ranges.append({"start": start, "end": cursor, "count": cursor - start})
    if len(ranges) != 1:
        raise RuntimeError(
            "qualification returned prompt must contain one contiguous image-pad "
            f"range, got {ranges}"
        )
    return ranges


def _close_engine(llm: Any | None, *, torch: Any, parent: Any) -> dict[str, Any]:
    children_before = _child_processes(parent)
    shutdown_called = False
    shutdown_error: str | None = None
    distributed_cleanup_called = False
    distributed_cleanup_error: str | None = None
    if llm is not None:
        try:
            engine = getattr(llm, "llm_engine", None)
            engine_core = getattr(engine, "engine_core", None)
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                shutdown()
                shutdown_called = True
            if engine is not None:
                engine.engine_core = None
            llm.llm_engine = None
        except Exception as exc:  # Cleanup evidence must survive the error.
            shutdown_error = f"{type(exc).__name__}: {exc}"
    try:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        distributed_cleanup_called = True
    except Exception as exc:
        distributed_cleanup_error = f"{type(exc).__name__}: {exc}"
    llm = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    deadline = time.monotonic() + 10.0
    children_after = _child_processes(parent)
    while children_after and time.monotonic() < deadline:
        time.sleep(0.1)
        children_after = _child_processes(parent)
    return {
        "shutdown_called": shutdown_called,
        "shutdown_error": shutdown_error,
        "distributed_cleanup_called": distributed_cleanup_called,
        "distributed_cleanup_error": distributed_cleanup_error,
        "owned_children_before_cleanup": children_before,
        "owned_children_after_cleanup": children_after,
        "cuda_cache_cleared": bool(torch.cuda.is_available()),
    }


def _validate_cuda(torch: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("qualification requires explicit CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        raise RuntimeError("qualification requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"qualification requires one visible logical GPU, got {torch.cuda.device_count()}"
        )
    if torch.cuda.current_device() != 0:
        raise RuntimeError("qualification worker must use logical cuda:0")


def _cuda_receipt(torch: Any) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "current_device": int(torch.cuda.current_device()) if available else None,
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "memory_allocated": int(torch.cuda.memory_allocated(0)) if available else 0,
        "memory_reserved": int(torch.cuda.memory_reserved(0)) if available else 0,
    }


def _child_processes(parent: Any) -> list[dict[str, Any]]:
    rows = []
    for child in parent.children(recursive=True):
        try:
            rows.append(
                {
                    "pid": int(child.pid),
                    "name": child.name(),
                    "status": child.status(),
                    "cmdline": child.cmdline(),
                }
            )
        except Exception:
            continue
    return sorted(rows, key=lambda row: row["pid"])


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _installed_file_identity(
    path: Path,
    *,
    package: str,
    package_root: Path,
) -> dict[str, Any]:
    path = path.resolve()
    package_root = package_root.resolve()
    return {
        "package": package,
        "relative_path": (Path(package) / path.relative_to(package_root)).as_posix(),
        **_file_identity(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(_jsonable(receipt), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
