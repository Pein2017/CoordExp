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
from uuid import UUID

from src.qwen.generation import ContinuationResult, ContinuationTrace, trim_suffix
from src.qwen.native import NativeRequest, _open_image


def _physical_identity(device):
    import torch
    properties = torch.cuda.get_device_properties(device)
    try:
        result = dict(uuid='GPU-'+str(UUID(str(properties.uuid).removeprefix('GPU-'))),
                      **{name: getattr(properties, name) for name in
                         ('pci_domain_id', 'pci_bus_id', 'pci_device_id')})
        if any(type(result[k]) is not int or result[k] < 0 for k in result if k != 'uuid'):
            raise ValueError('invalid PCI identity')
        return result
    except (AttributeError, ValueError, TypeError) as exc:
        raise RuntimeError('required CUDA UUID/PCI identity unavailable') from exc


def _process_identity():
    return dict(pid=os.getpid(), ppid=os.getppid(),
                nspid=next(line for line in Path('/proc/self/status').read_text().splitlines()
                           if line.startswith('NSpid:')))


def _physical_token(visibility, device):
    if type(device) is not int or device < 0:
        raise ValueError('invalid requested CUDA ordinal')
    if visibility is None:
        return str(device)
    tokens = visibility.split(',')
    if device >= len(tokens) or any(not t or t != t.strip() for t in tokens):
        raise ValueError('requested CUDA ordinal absent from visibility mask')
    token = tokens[device]
    if not (token.isdecimal() or token.startswith('GPU-')):
        raise ValueError('unsupported CUDA visibility token')
    return token


def _check_physical(physical, token):
    if not isinstance(physical, dict) or set(physical) != {'uuid', 'pci_domain_id', 'pci_bus_id', 'pci_device_id'}:
        raise ValueError('missing physical device identity')
    if physical['uuid'] != 'GPU-'+str(UUID(physical['uuid'].removeprefix('GPU-'))):
        raise ValueError('invalid GPU UUID identity')
    if any(type(physical[k]) is not int or physical[k] < 0 for k in physical if k != 'uuid'):
        raise ValueError('invalid PCI identity')
    if token.startswith('GPU-') and not physical['uuid'].lower().startswith(token.lower()):
        raise ValueError('visibility UUID differs from selected parent device')


def _check_process(process):
    if type(process.get('pid')) is not int or type(process.get('ppid')) is not int or min(process['pid'], process['ppid']) <= 0:
        raise ValueError('missing process identity')
    raw = process.get('nspid')
    if not isinstance(raw, str) or not raw.startswith('NSpid:') or int(raw.split()[-1]) != process['pid']:
        raise ValueError('missing raw NSpid identity')


def validate_device_receipt(receipt, request):
    """Validate persisted admission evidence without querying a device."""
    try:
        if request['schema'] != 'coordexp-vllm-device-1' or type(request['rank']) is not int or request['rank'] < 0:
            raise ValueError('invalid requested rank')
        parent = request['parent']; token = request['physical_token']
        _check_process(parent)
        if token != _physical_token(parent['visibility'], request['device']):
            raise ValueError('requested visibility association differs')
        _check_physical(parent['physical'], token)
        if receipt['requested'] != request:
            raise ValueError('child request differs from parent')
        _check_process(receipt['child'])
        if receipt['child']['ppid'] != parent['pid']:
            raise ValueError('child parent PID differs')
        if receipt['inherited_visibility'] != token or receipt['effective_visibility'] != token or type(receipt['logical_device']) is not int or receipt['logical_device'] != 0:
            raise ValueError('child visibility/logical device differs')
        _check_physical(receipt['physical'], token)
        if receipt['physical'] != parent['physical']:
            raise ValueError('child physical device differs from parent')
    except (KeyError, TypeError, AttributeError, ValueError, IndexError) as exc:
        raise ValueError(f'invalid vLLM device receipt: {exc}') from exc


def validate_device_assignments(rows, devices):
    if len(rows) != len(devices):
        raise ValueError('missing rank device evidence')
    try:
        for rank, (row, device) in enumerate(zip(rows, devices, strict=True)):
            request = row['request']
            if type(row['rank']) is not int or row['rank'] != rank or request['rank'] != rank or request['device'] != device:
                raise ValueError('reordered or mismatched rank/device association')
            validate_device_receipt(row['startup']['device'], request)
    except (KeyError, TypeError) as exc:
        raise ValueError('missing rank device evidence') from exc
    physical = [row['request']['parent']['physical'] for row in rows]
    if len({x['uuid'] for x in physical}) != len(rows) or len({(x['pci_domain_id'], x['pci_bus_id'], x['pci_device_id']) for x in physical}) != len(rows):
        raise ValueError('duplicate physical vLLM device')


def _child_device_receipt(request):
    import torch
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited != request['physical_token'] or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise ValueError('child CUDA visibility/logical device admission failed')
    receipt = dict(requested=request, child=_process_identity(), inherited_visibility=inherited,
                   effective_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), logical_device=0,
                   physical=_physical_identity(0))
    validate_device_receipt(receipt, request)
    return receipt


def _refresh_model(model, adapter, embeddings, identity):
    model.refresh_coordexp_dora(adapter, embeddings, identity=identity)
    return identity


def _configure_coordinate_output_norm(model, mode, token_ids, identity):
    return model.configure_coordinate_output_norm(mode, token_ids, identity=identity)


def _coordinate_output_norm_receipt(model):
    return model.coordinate_output_norm_receipt()


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


def _worker(connection, request, base_model, checkpoint, identity, options, log_path):
    # A separate process avoids sharing vLLM's process group with training DDP.
    for key in list(os.environ):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                   "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                   "ROLE_WORLD_SIZE"} or key.startswith("TORCHELASTIC_"):
            os.environ.pop(key)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    engine = None
    try:
        with open(log_path, "a", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            device_receipt = _child_device_receipt(request)
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
            connection.send({"ok": True, "value": {"identity": identity, "device": device_receipt,
                                                     "startup_seconds": time.monotonic()-started}})
            norm_configured = False
            while True:
                command, payload = connection.recv()
                if command == "close":
                    break
                started = time.monotonic()
                if command == "coordinate_output_norm":
                    mode, token_ids, requested_identity = payload
                    if requested_identity != identity:
                        raise ValueError("stale coordinate norm snapshot identity")
                    observed = engine.apply_model(partial(_configure_coordinate_output_norm,
                        mode=mode, token_ids=token_ids, identity=identity))
                    if len(observed) != 1 or observed[0]['mode'] != mode or observed[0]['identity'] != identity:
                        raise RuntimeError("coordinate output norm was not acknowledged")
                    value = observed[0]
                    norm_configured = True
                elif command == "refresh":
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
                norm_evidence = (engine.apply_model(_coordinate_output_norm_receipt)[0]
                                 if norm_configured else None)
                connection.send({"ok": True, "value": value,
                                 "receipt": {"identity": identity,
                                             "seconds": time.monotonic()-started,
                                             "peak_allocated": torch.cuda.max_memory_allocated(),
                                             **({"coordinate_output_norm": norm_evidence} if norm_configured else {})}})
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
                 device=None, trainer_rank=None, max_model_len=16000, max_num_seqs=3,
                 kv_cache_memory_bytes=2 * 1024**3, gpu_memory_utilization=0.2,
                 enforce_eager=False, timeout=1800, seed=None):
        import torch
        from importlib.metadata import version
        if version("vllm").split("+")[0] != "0.29.0":
            raise RuntimeError("local DoRA rollout is qualified only for vLLM 0.29.0")
        if not isinstance(identity, str) or not identity:
            raise ValueError("a nonempty snapshot identity is required")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('vLLM seed must be an unsigned 32-bit integer')
        device_index = torch.cuda.current_device() if device is None else device
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_device = _physical_token(visible, device_index)
        self.device_request = dict(schema='coordexp-vllm-device-1',
            rank=int(os.environ.get('RANK', '0')) if trainer_rank is None else trainer_rank,
            device=device_index, physical_token=physical_device,
            parent=dict(_process_identity(), visibility=visible, physical=_physical_identity(device_index)))
        _check_physical(self.device_request['parent']['physical'], physical_device)
        log_path = Path(log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._timeout, self._closed = timeout, False
        self.identity = identity
        self.receipts = []
        options = dict(max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                       kv_cache_memory_bytes=kv_cache_memory_bytes,
                       gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager)
        if seed is not None:
            options['seed'] = seed
        if not enforce_eager:
            # Native decode graphs keep refreshable buffers; no compiler needed.
            options['compilation_config'] = dict(mode=0, cudagraph_mode='FULL_DECODE_ONLY',
                                                 cudagraph_capture_sizes=list(range(1, max_num_seqs+1)))
        self._process = context.Process(target=_worker, args=(
            child, self.device_request, str(Path(base_model).resolve()),
            str(Path(checkpoint).resolve()), identity,
            options,
            str(log_path)), name="coordexp-vllm-rollout")
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = physical_device
            self._process.start()
        except BaseException:
            child.close()
            self._connection.close()
            self._closed = True
            raise
        finally:
            if visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
        child.close()
        atexit.register(self.close)
        try:
            self.startup = self._receive()["value"]
            validate_device_receipt(self.startup['device'], self.device_request)
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

    def configure_coordinate_output_norm(self, mode, token_ids, *, identity):
        if mode not in ('off', 'median') or identity != self.identity:
            raise ValueError('invalid coordinate norm policy or stale snapshot identity')
        return self._call('coordinate_output_norm', (mode, list(token_ids), identity))

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
