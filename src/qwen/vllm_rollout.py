"""Resident, TP=1 DoRA rollout engine isolated from the HF training process.

The caller owns training, snapshot identity and sampling scope. Only empty-history
greedy continuations are supported. No model merge, checkpoint export or package
patch is performed when refreshing the current adapter.
"""
from __future__ import annotations

import atexit
from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from src.qwen.generation import ContinuationResult, ContinuationTrace, trim_suffix
from src.qwen.native import NativeRequest, _open_image


def _refresh_model(model, adapter, embeddings, identity):
    model.refresh_coordexp_dora(adapter, embeddings, identity=identity)
    return identity


def _generate(engine, requests, budgets, eos_token_id, pad_token_id, trace):
    from vllm import SamplingParams
    from vllm.inputs import TextPrompt

    if len(requests) != len(budgets) or not requests:
        raise ValueError("requests and budgets must be nonempty and aligned")
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("duplicate rollout request IDs")
    if any(type(b) is not int or b <= 0 for b in budgets):
        raise ValueError("rollout budgets must be positive integers")
    if any(r.expected_token_ids is None for r in requests):
        raise ValueError("rollout requires authoritative HF prompt token IDs")
    images = []
    try:
        for request in requests:
            images.append(_open_image(request))
        prompts = [TextPrompt(prompt=r.chat_text, multi_modal_data={"image": im},
                              mm_processor_kwargs={"do_resize": False})
                   for r, im in zip(requests, images, strict=True)]
        params = [SamplingParams(temperature=0, top_p=1, top_k=-1,
                                 repetition_penalty=1, max_tokens=b,
                                 stop_token_ids=[eos_token_id], detokenize=False,
                                 logprobs=1 if trace else None)
                  for b in budgets]
        outputs = engine.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM omitted rollout requests")
        results = []
        for request, budget, output in zip(requests, budgets, outputs, strict=True):
            if tuple(output.prompt_token_ids) != request.expected_token_ids:
                raise RuntimeError(f"vLLM changed prompt tokens: {request.request_id}")
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM must return exactly one continuation")
            completion = output.outputs[0]
            ids, reason = trim_suffix(completion.token_ids, budget=budget,
                                      eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            if completion.finish_reason != ("stop" if reason == "im_end" else "length"):
                raise RuntimeError("vLLM stop reason differs from token evidence")
            evidence = None
            if trace:
                import math
                scores = tuple(float(row[token].logprob) for token, row in
                               zip(ids, completion.logprobs, strict=True))
                if not all(math.isfinite(x) for x in scores):
                    raise RuntimeError("nonfinite vLLM raw log probabilities")
                evidence = ContinuationTrace(ids, scores, scores)
            results.append(ContinuationResult(request.request_id, ids, reason, evidence))
        return tuple(results)
    finally:
        for image in images:
            image.close()


def _worker(connection, device, base_model, checkpoint, identity, options, log_path):
    # A separate process avoids sharing vLLM's process group with training DDP.
    for key in list(os.environ):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                   "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                   "ROLE_WORLD_SIZE"} or key.startswith("TORCHELASTIC_"):
            os.environ.pop(key)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    engine = None
    try:
        with open(log_path, "a", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            from vllm import LLM, ModelRegistry
            ModelRegistry.register_model(
                "CoordExpDoRAQwen3VLForConditionalGeneration",
                "src.qwen.vllm_dora_model:CoordExpDoRAQwen3VLForConditionalGeneration",
            )
            started = time.monotonic()
            engine = LLM(
                model=base_model, tokenizer=base_model, dtype="bfloat16",
                tensor_parallel_size=1, pipeline_parallel_size=1,
                distributed_executor_backend="uni", generation_config="vllm",
                logprobs_mode="raw_logprobs", enable_prefix_caching=False,
                hf_overrides={"architectures": ["CoordExpDoRAQwen3VLForConditionalGeneration"],
                              "coordexp_dora": {
                                  "adapter_path": str(Path(checkpoint) / "adapter"),
                                  "embedding_path": str(Path(checkpoint) / "special_token_embeddings"),
                                  "identity": identity}},
                **options,
            )
            connection.send({"ok": True, "value": {"identity": identity,
                                                     "startup_seconds": time.monotonic()-started}})
            while True:
                command, payload = connection.recv()
                if command == "close":
                    break
                started = time.monotonic()
                if command == "refresh":
                    from safetensors.torch import load
                    adapter_bytes, embedding_bytes, next_identity = payload
                    adapter, embeddings = load(adapter_bytes), load(embedding_bytes)
                    installed = engine.apply_model(partial(_refresh_model, adapter=adapter,
                                                           embeddings=embeddings, identity=next_identity))
                    if installed != [next_identity]:
                        raise RuntimeError("vLLM adapter refresh was not acknowledged")
                    if not engine.reset_prefix_cache():
                        raise RuntimeError("vLLM did not reset its prefix cache")
                    identity = next_identity
                    value = None
                elif command == "generate":
                    requested_identity, *generation = payload
                    if requested_identity != identity:
                        raise ValueError("stale rollout snapshot identity")
                    value = _generate(engine, *generation)
                else:
                    raise ValueError(f"unknown rollout operation: {command}")
                import torch
                connection.send({"ok": True, "value": value,
                                 "receipt": {"identity": identity,
                                             "seconds": time.monotonic()-started,
                                             "peak_allocated": torch.cuda.max_memory_allocated()}})
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if engine is not None:
            from src.inference.vllm_backend import _close_vllm_engine
            _close_vllm_engine(engine)
        connection.close()


class VllmDoraRollout:
    """One synchronous engine per trainer GPU; no concurrent refresh/generation."""

    def __init__(self, *, base_model, checkpoint, identity, log_path,
                 device=None, max_model_len=16000, max_num_seqs=3,
                 kv_cache_memory_bytes=2 * 1024**3, enforce_eager=True, timeout=1800):
        import torch
        from importlib.metadata import version
        if version("vllm").split("+")[0] != "0.29.0":
            raise RuntimeError("local DoRA rollout is qualified only for vLLM 0.29.0")
        if not isinstance(identity, str) or not identity:
            raise ValueError("a nonempty snapshot identity is required")
        device_index = torch.cuda.current_device() if device is None else device
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_device = visible.split(",")[device_index] if visible else str(device_index)
        log_path = Path(log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._timeout, self._closed = timeout, False
        self.identity = identity
        self.receipts = []
        self._process = context.Process(target=_worker, args=(
            child, physical_device, str(Path(base_model).resolve()),
            str(Path(checkpoint).resolve()), identity,
            dict(max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                 kv_cache_memory_bytes=kv_cache_memory_bytes, enforce_eager=enforce_eager),
            str(log_path)), name="coordexp-vllm-rollout")
        self._process.start()
        child.close()
        atexit.register(self.close)
        try:
            self.startup = self._receive()["value"]
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self._connection.poll(self._timeout):
            raise TimeoutError("vLLM rollout operation timed out; inspect worker log")
        response = self._connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response

    def _call(self, command, payload):
        if self._closed:
            raise RuntimeError("rollout engine is closed")
        try:
            self._connection.send((command, payload))
            response = self._receive()
            self.receipts.append(dict(operation=command, **response["receipt"]))
            return response["value"]
        except BaseException:
            self.close()
            raise

    def refresh(self, model, embedding_deltas, *, identity):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save
        if not isinstance(identity, str) or not identity:
            raise ValueError("a nonempty snapshot identity is required")
        adapter = {k: v.detach().to(device="cpu", copy=True) for k, v in
                   get_peft_model_state_dict(model, adapter_name="default").items()}
        embeddings = {k: v.detach().to(device="cpu", copy=True) for k, v in
                      embedding_deltas.delta_tensors().items()}
        # One byte message avoids a shared-memory file descriptor per tensor.
        self._call("refresh", (save(adapter), save(embeddings), identity))
        self.identity = identity

    def generate(self, requests: list[NativeRequest], *, budgets, eos_token_id,
                 pad_token_id, identity, trace=False):
        return self._call("generate", (identity, requests, budgets, eos_token_id,
                                       pad_token_id, trace))

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            self._connection.send(("close", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._connection.close()
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
