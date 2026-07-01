from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DetectionDecodeResult:
    text: str
    generated_token_ids: Optional[list[int]]
    generated_tokens: Optional[list[str]]
    generated_logprobs: Optional[list[float]]
    stop_reason: Optional[str]
    backend: str
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: Optional[list[int]] = None


@dataclass(frozen=True)
class HFRolloutBackendHandles:
    """Lifecycle handles needed by the HF rollout backend core."""

    template: Any
    model: Any
    model_wrapped: Any
    accelerator: Any
    ds3_gather_for_generation: bool
    decode_batch_size: int
    offload_context_fn: Callable[..., Any]
    template_packing_disabled_fn: Callable[[], Any]
    rollout_template_policy: Any
    unwrap_model_for_generation_fn: Callable[..., Any]
    num_return_sequences: int


def resolve_hf_rollout_backend_handles_from_owner(
    *,
    owner: Any,
    unwrap_model_for_generation_fn: Callable[..., Any],
) -> HFRolloutBackendHandles:
    """Translate a Stage-2 owner into explicit HF rollout backend handles."""

    from src.infer.runtime import resolve_rollout_runtime_facts_from_owner, rollout_owner_cfg

    rollout_facts = resolve_rollout_runtime_facts_from_owner(
        owner,
        rollout_backend="hf",
    )
    num_beams = max(1, int(rollout_owner_cfg(owner, "num_beams", 1)))
    return HFRolloutBackendHandles(
        template=owner.template,
        model=owner.model,
        model_wrapped=owner.model_wrapped,
        accelerator=owner.accelerator,
        ds3_gather_for_generation=bool(
            getattr(owner.args, "ds3_gather_for_generation", False)
        ),
        decode_batch_size=int(rollout_facts.decode_batch_size),
        offload_context_fn=owner._maybe_rollout_offload_context,
        template_packing_disabled_fn=owner._template_packing_disabled,
        rollout_template_policy=owner._eval_rollout_template_policy(),
        unwrap_model_for_generation_fn=unwrap_model_for_generation_fn,
        num_return_sequences=max(
            1,
            int(
                rollout_owner_cfg(
                    owner,
                    "num_return_sequences",
                    int(num_beams),
                )
            ),
        ),
    )


@lru_cache(maxsize=2)
def vllm_engine_args_fields(*, async_engine: bool = False) -> frozenset[str]:
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for rollout_matching.vllm compatibility checks. "
            "Use rollout_backend=hf to bypass vLLM."
        ) from exc

    cls = AsyncEngineArgs if bool(async_engine) else EngineArgs
    fields = set(getattr(cls, "__dataclass_fields__", {}) or {})
    fields.update(inspect.signature(cls).parameters)
    return frozenset(str(name) for name in fields)


def validate_vllm_engine_kwargs(
    kwargs: Mapping[str, Any],
    *,
    context: str,
    async_engine: bool = False,
    required: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    supported = vllm_engine_args_fields(async_engine=async_engine)
    out = dict(kwargs)

    unsupported = sorted(str(key) for key in out if str(key) not in supported)
    if unsupported:
        raise RuntimeError(
            f"{context} contains vLLM EngineArgs keys unsupported by the installed "
            f"vLLM package: {unsupported}. Remove the config key or upgrade to a "
            "compatible vLLM build before launching server-mode rollouts."
        )

    missing_required = sorted(str(key) for key in required if str(key) not in supported)
    if missing_required:
        raise RuntimeError(
            f"{context} requires vLLM EngineArgs keys unsupported by the installed "
            f"vLLM package: {missing_required}. Upgrade vLLM or change rollout backend."
        )

    return out


def vllm_request_config_kwargs_from_decode_request(request: Any) -> dict[str, Any]:
    """Project a shared decode request into ms-swift/vLLM RequestConfig kwargs."""

    stop_strings = getattr(request, "stop_strings", None) or ("<|im_end|>",)
    return {
        "n": 1,
        "max_tokens": int(getattr(request, "max_new_tokens")),
        "temperature": float(getattr(request, "temperature")),
        "top_p": float(getattr(request, "top_p")),
        "top_k": int(getattr(request, "top_k")),
        "repetition_penalty": float(getattr(request, "repetition_penalty")),
        "stop": [str(stop) for stop in stop_strings],
        "return_details": True,
    }


def build_swift_request_config_from_decode_request(
    request: Any,
    *,
    seed: int | None = None,
    trace_logprobs: bool = False,
) -> Any:
    """Build ms-swift RequestConfig from a shared decode request plus overlays."""

    request_kwargs = vllm_request_config_kwargs_from_decode_request(request)
    if seed is not None:
        request_kwargs["seed"] = int(seed)
    if trace_logprobs:
        request_kwargs["logprobs"] = True
    RequestConfig = import_swift_request_config()
    return RequestConfig(**request_kwargs)


def apply_hf_generation_config_from_decode_request(
    *,
    gen_cfg: Any,
    request: Any,
) -> Any:
    """Project shared decode sampling knobs into a mutable HF GenerationConfig."""

    temperature = float(getattr(request, "temperature"))
    do_sample = bool(temperature > 0.0)
    top_k = int(getattr(request, "top_k"))

    gen_cfg.max_new_tokens = int(getattr(request, "max_new_tokens"))
    gen_cfg.do_sample = do_sample
    gen_cfg.temperature = max(1e-4, temperature) if do_sample else 1.0
    gen_cfg.top_p = float(getattr(request, "top_p")) if do_sample else 1.0
    gen_cfg.top_k = top_k if (do_sample and top_k != -1) else 0
    gen_cfg.repetition_penalty = float(getattr(request, "repetition_penalty"))
    gen_cfg.use_cache = True
    return gen_cfg


def enforce_hf_rollout_max_position_embeddings(
    *,
    model: Any,
    prompt_pad_len: int,
    max_new_tokens: int,
) -> None:
    """Fail fast when HF rollout prompt+generation exceeds model context."""

    cfg = getattr(model, "config", None)
    max_pos_raw = getattr(cfg, "max_position_embeddings", None)
    if max_pos_raw is None:
        return
    try:
        max_pos = int(max_pos_raw)
    except (TypeError, ValueError):
        return
    if max_pos <= 0:
        return

    needed = int(prompt_pad_len) + int(max_new_tokens)
    if needed <= max_pos:
        return

    raise ValueError(
        "HF rollout would exceed model.max_position_embeddings: "
        f"prompt_pad_len={int(prompt_pad_len)} max_new_tokens={int(max_new_tokens)} "
        f"needed={int(needed)} max_position_embeddings={int(max_pos)}. "
        "Reduce rollout_matching.max_new_tokens and/or ensure prompts fit within the model context."
    )


def build_hf_rollout_logits_processor(
    *,
    tokenizer: Any,
    prompt_pad_len: int,
    batch_size: int,
    rollout_template_policy: Any,
    trailing_processors: Optional[Sequence[Any]] = None,
) -> Any:
    """Build retained non-grammar HF logits processors for rollout decoding."""

    _ = tokenizer, prompt_pad_len, batch_size, rollout_template_policy
    processors: List[Any] = []
    if trailing_processors:
        processors.extend(trailing_processors)
    if not processors:
        return None

    from transformers import LogitsProcessorList

    return LogitsProcessorList(processors)


def _prompt_token_ids_from_batch(input_ids_t: Any, attention_mask: Any) -> List[List[int]]:
    prompt_ids_list: List[List[int]] = []
    for row_ids, row_mask in zip(input_ids_t, attention_mask):
        ids = [
            int(token_id)
            for token_id, mask_value in zip(
                row_ids.detach().cpu().tolist(),
                row_mask.detach().cpu().tolist(),
            )
            if int(mask_value) == 1
        ]
        prompt_ids_list.append(ids)
    return prompt_ids_list


def _hf_rollout_batch_inputs(
    *,
    template: Any,
    samples: Sequence[Mapping[str, Any]],
    to_device: Callable[[Any, Any], Any],
    device: Any,
) -> tuple[Any, Any, List[List[int]]]:
    import torch

    with template.generate_context():
        encoded_list = [template.encode(dict(sample), return_length=True) for sample in samples]
        batch = template.data_collator(encoded_list)

    batch = to_device(batch, device)
    input_ids_t = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        pad_id = int(getattr(template.tokenizer, "pad_token_id", 0) or 0)
        attention_mask = (input_ids_t != pad_id).to(dtype=torch.long)
    prompt_ids_list = _prompt_token_ids_from_batch(input_ids_t, attention_mask)
    return batch, input_ids_t, prompt_ids_list


def rollout_many_hf(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    decode_request: Any,
    unwrap_model_for_generation_fn: Callable[..., Any],
) -> List[Tuple[List[int], str, str, List[int]]]:
    """Owner adapter for the HF rollout backend."""

    return rollout_many_hf_with_handles(
        handles=resolve_hf_rollout_backend_handles_from_owner(
            owner=owner,
            unwrap_model_for_generation_fn=unwrap_model_for_generation_fn,
        ),
        samples=samples,
        decode_request=decode_request,
    )


def rollout_many_hf_with_handles(
    *,
    handles: HFRolloutBackendHandles,
    samples: Sequence[Mapping[str, Any]],
    decode_request: Any,
) -> List[Tuple[List[int], str, str, List[int]]]:
    """HF rollout backend using the shared inference backend implementation."""

    template = handles.template
    tokenizer = template.tokenizer
    decode_mode = str(decode_request.decode_mode)
    max_new_tokens = int(decode_request.max_new_tokens)
    num_beams = int(decode_request.num_beams)

    gen_cfg = getattr(handles.model, "generation_config", None)
    if gen_cfg is None:
        from transformers import GenerationConfig

        gen_cfg = GenerationConfig()
    gen_cfg = deepcopy(gen_cfg)
    apply_hf_generation_config_from_decode_request(gen_cfg=gen_cfg, request=decode_request)
    if decode_mode == "beam":
        gen_cfg.num_beams = max(1, num_beams)
        gen_cfg.num_return_sequences = max(
            1,
            int(handles.num_return_sequences),
        )
    else:
        gen_cfg.num_beams = 1
        gen_cfg.num_return_sequences = 1

    from src.common.qwen_generation import resolve_qwen_chat_generation_token_ids

    qwen_generation_ids = resolve_qwen_chat_generation_token_ids(tokenizer)
    gen_cfg.eos_token_id = qwen_generation_ids.eos_token_id
    gen_cfg.pad_token_id = qwen_generation_ids.pad_token_id

    out: List[Tuple[List[int], str, str, List[int]]] = []
    microbatch = int(handles.decode_batch_size)
    to_device = import_swift_to_device()

    with handles.offload_context_fn(rollout_backend="hf"):
        idx = 0
        while idx < len(samples):
            chunk = list(samples[idx : idx + microbatch])
            idx += len(chunk)

            with handles.template_packing_disabled_fn():
                batch, input_ids_t, prompt_ids_list = _hf_rollout_batch_inputs(
                    template=template,
                    samples=chunk,
                    to_device=to_device,
                    device=handles.model.device,
                )

            prompt_pad_len = int(input_ids_t.shape[1])
            enforce_hf_rollout_max_position_embeddings(
                model=handles.model,
                prompt_pad_len=prompt_pad_len,
                max_new_tokens=max_new_tokens,
            )
            model_inputs = {key: value for key, value in batch.items() if key != "labels"}
            model_inputs.pop("position_ids", None)
            model_inputs.pop("text_position_ids", None)

            logits_processor = build_hf_rollout_logits_processor(
                tokenizer=tokenizer,
                prompt_pad_len=prompt_pad_len,
                batch_size=int(input_ids_t.shape[0]),
                rollout_template_policy=handles.rollout_template_policy,
            )

            with handles.unwrap_model_for_generation_fn(
                handles.model_wrapped,
                handles.accelerator,
                gather_deepspeed3_params=handles.ds3_gather_for_generation,
            ) as unwrapped:
                unwrapped.eval()
                with handles.template_packing_disabled_fn():
                    with template.generate_context():
                        if (
                            getattr(handles.model, "model_meta", None) is not None
                            and handles.model.model_meta.is_multimodal
                        ):
                            _, model_inputs = template.pre_forward_hook(
                                unwrapped,
                                None,
                                model_inputs,
                            )
                        model_inputs.pop("position_ids", None)
                        model_inputs.pop("text_position_ids", None)
                        gen_out = template.generate(
                            unwrapped,
                            **model_inputs,
                            generation_config=gen_cfg,
                            return_dict_in_generate=True,
                            logits_processor=logits_processor,
                        )
                unwrapped.train()

            sequences = gen_out.sequences
            if sequences.ndim != 2:
                raise ValueError("unexpected generate output shape")

            batch_size = int(input_ids_t.shape[0])
            nret = int(getattr(gen_cfg, "num_return_sequences", 1) or 1)
            if nret < 1:
                nret = 1
            if (
                decode_mode == "beam"
                and nret > 1
                and hasattr(gen_out, "sequences_scores")
                and gen_out.sequences_scores is not None
            ):
                scores = gen_out.sequences_scores
                if scores.ndim != 1 or sequences.shape[0] != batch_size * nret:
                    import torch

                    best_idx = torch.zeros(
                        (batch_size,),
                        dtype=torch.long,
                        device=sequences.device,
                    )
                else:
                    scores = scores.view(batch_size, nret)
                    best_idx = scores.argmax(dim=1)
                    import torch
                sequences = sequences.view(batch_size, nret, -1)
                best_seqs = sequences[
                    torch.arange(batch_size, device=sequences.device),
                    best_idx,
                ]
            else:
                if sequences.shape[0] == batch_size * nret:
                    sequences = sequences.view(batch_size, nret, -1)[:, 0, :]
                else:
                    sequences = sequences[:batch_size, :]
                best_seqs = sequences

            for sample_index in range(batch_size):
                seq = best_seqs[sample_index]
                resp_ids = seq[prompt_pad_len:].tolist()
                resp_ids = template.skip_stop_tokens(resp_ids, is_finished=True)
                text = template.decode(
                    resp_ids,
                    is_finished=True,
                    first_token=True,
                    clean_up_tokenization_spaces=False,
                )
                out.append((resp_ids, text, decode_mode, prompt_ids_list[sample_index]))

    return out


def rollout_many_hf_traced(
    *,
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    decode_request: Any,
    unwrap_model_for_generation_fn: Callable[..., Any],
) -> List[Tuple[List[int], str, str, List[int], List[float], List[str]]]:
    """Owner adapter for traced HF rollout generation."""

    return rollout_many_hf_traced_with_handles(
        handles=resolve_hf_rollout_backend_handles_from_owner(
            owner=owner,
            unwrap_model_for_generation_fn=unwrap_model_for_generation_fn,
        ),
        samples=samples,
        decode_request=decode_request,
    )


def rollout_many_hf_traced_with_handles(
    *,
    handles: HFRolloutBackendHandles,
    samples: Sequence[Mapping[str, Any]],
    decode_request: Any,
) -> List[Tuple[List[int], str, str, List[int], List[float], List[str]]]:
    """HF rollout backend with strict generated-token logprob tracing."""

    import torch

    template = handles.template
    tokenizer = template.tokenizer
    decode_mode = str(decode_request.decode_mode)
    if decode_mode == "beam":
        raise ValueError("eval-step confidence scoring does not support decode_mode=beam")
    if float(decode_request.temperature) > 0.0:
        raise ValueError(
            "eval-step confidence scoring requires decoding.temperature=0.0 "
            f"(greedy), got {float(decode_request.temperature)}"
        )

    gen_cfg = getattr(handles.model, "generation_config", None)
    if gen_cfg is None:
        from transformers import GenerationConfig

        gen_cfg = GenerationConfig()
    gen_cfg = deepcopy(gen_cfg)
    apply_hf_generation_config_from_decode_request(gen_cfg=gen_cfg, request=decode_request)
    gen_cfg.num_beams = 1
    gen_cfg.num_return_sequences = 1

    from src.common.qwen_generation import resolve_qwen_chat_generation_token_ids

    qwen_generation_ids = resolve_qwen_chat_generation_token_ids(tokenizer)
    gen_cfg.eos_token_id = qwen_generation_ids.eos_token_id
    gen_cfg.pad_token_id = qwen_generation_ids.pad_token_id

    try:
        from transformers.generation.logits_process import (
            LogitsProcessor,
            LogitsProcessorList,
        )
    except Exception:  # pragma: no cover
        from transformers.generation_logits_process import (  # type: ignore[no-redef]
            LogitsProcessor,
            LogitsProcessorList,
        )

    class _GreedyTokenLogprobTracer(LogitsProcessor):
        def __init__(self) -> None:
            self.token_logprobs: List[List[float]] = []

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if not self.token_logprobs:
                batch_size = int(scores.shape[0])
                self.token_logprobs = [[] for _ in range(batch_size)]

            token_ids = torch.argmax(scores, dim=-1)
            scores_f = scores.float()
            selected = scores_f.gather(dim=1, index=token_ids.unsqueeze(1)).squeeze(1)
            log_norm = torch.logsumexp(scores_f, dim=-1)
            logprobs = (selected - log_norm).detach().cpu().tolist()
            for index, logprob in enumerate(logprobs):
                self.token_logprobs[index].append(float(logprob))
            return scores

    out: List[Tuple[List[int], str, str, List[int], List[float], List[str]]] = []
    microbatch = int(handles.decode_batch_size)
    to_device = import_swift_to_device()

    idx = 0
    while idx < len(samples):
        chunk = list(samples[idx : idx + microbatch])
        idx += len(chunk)

        with handles.template_packing_disabled_fn():
            batch, input_ids_t, prompt_ids_list = _hf_rollout_batch_inputs(
                template=template,
                samples=chunk,
                to_device=to_device,
                device=handles.model.device,
            )

        prompt_pad_len = int(input_ids_t.shape[1])
        enforce_hf_rollout_max_position_embeddings(
            model=handles.model,
            prompt_pad_len=prompt_pad_len,
            max_new_tokens=int(decode_request.max_new_tokens),
        )
        model_inputs = {key: value for key, value in batch.items() if key != "labels"}
        model_inputs.pop("position_ids", None)
        model_inputs.pop("text_position_ids", None)

        tracer = _GreedyTokenLogprobTracer()
        logits_processor = build_hf_rollout_logits_processor(
            tokenizer=tokenizer,
            prompt_pad_len=prompt_pad_len,
            batch_size=int(input_ids_t.shape[0]),
            rollout_template_policy=handles.rollout_template_policy,
            trailing_processors=[tracer],
        )
        if logits_processor is None:
            logits_processor = LogitsProcessorList([tracer])

        with handles.unwrap_model_for_generation_fn(
            handles.model_wrapped,
            handles.accelerator,
            gather_deepspeed3_params=handles.ds3_gather_for_generation,
        ) as unwrapped:
            unwrapped.eval()
            with handles.template_packing_disabled_fn():
                with template.generate_context():
                    if (
                        getattr(handles.model, "model_meta", None) is not None
                        and handles.model.model_meta.is_multimodal
                    ):
                        _, model_inputs = template.pre_forward_hook(
                            unwrapped,
                            None,
                            model_inputs,
                        )
                    model_inputs.pop("position_ids", None)
                    model_inputs.pop("text_position_ids", None)
                    gen_out = template.generate(
                        unwrapped,
                        **model_inputs,
                        generation_config=gen_cfg,
                        return_dict_in_generate=True,
                        logits_processor=logits_processor,
                    )
            unwrapped.train()

        sequences = gen_out.sequences
        if sequences.ndim != 2:
            raise ValueError("unexpected generate output shape")

        batch_size = int(input_ids_t.shape[0])
        sequences = sequences[:batch_size, :]
        for sample_index in range(batch_size):
            seq = sequences[sample_index]
            resp_ids_full = [int(token_id) for token_id in seq[prompt_pad_len:].tolist()]
            resp_ids = template.skip_stop_tokens(resp_ids_full, is_finished=True)

            token_logprobs_full = tracer.token_logprobs[sample_index]
            if len(token_logprobs_full) < len(resp_ids):
                raise RuntimeError(
                    "rollout logprob trace shorter than generated token ids: "
                    f"trace_len={len(token_logprobs_full)} gen_len={len(resp_ids)}"
                )
            token_logprobs = [float(value) for value in token_logprobs_full[: len(resp_ids)]]
            generated_token_text = decode_token_pieces_with_tokenizer(
                tokenizer=tokenizer,
                token_ids=resp_ids,
            )
            if len(generated_token_text) != len(token_logprobs):
                raise RuntimeError(
                    "rollout trace token/text length mismatch: "
                    f"text_len={len(generated_token_text)} logprob_len={len(token_logprobs)}"
                )

            text = template.decode(
                resp_ids,
                is_finished=True,
                first_token=True,
                clean_up_tokenization_spaces=False,
            )
            out.append(
                (
                    resp_ids,
                    text,
                    decode_mode,
                    prompt_ids_list[sample_index],
                    token_logprobs,
                    generated_token_text,
                )
            )

    return out


def import_swift_request_config() -> Any:
    """Import ms-swift RequestConfig across pre/post infer_engine API layouts."""

    errors: list[BaseException] = []
    for module_name in (
        "swift.infer_engine",
        "swift.infer_engine.protocol",
        "swift.llm",
    ):
        try:
            module = __import__(module_name, fromlist=("RequestConfig",))
            return getattr(module, "RequestConfig")
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "ms-swift RequestConfig is required for vLLM rollouts; tried "
        "swift.infer_engine.RequestConfig, swift.infer_engine.protocol.RequestConfig, "
        "and swift.llm.RequestConfig"
    ) from errors[-1]


def import_swift_infer_request_and_config() -> tuple[Any, Any]:
    """Import ms-swift InferRequest/RequestConfig across infer API layouts."""

    errors: list[BaseException] = []
    for module_name in (
        "swift.infer_engine",
        "swift.infer_engine.protocol",
        "swift.llm",
    ):
        try:
            module = __import__(
                module_name,
                fromlist=("InferRequest", "RequestConfig"),
            )
            return getattr(module, "InferRequest"), getattr(module, "RequestConfig")
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "ms-swift InferRequest and RequestConfig are required for vLLM rollouts; "
        "tried swift.infer_engine, swift.infer_engine.protocol, and swift.llm"
    ) from errors[-1]


def import_swift_to_device() -> Any:
    """Import ms-swift to_device across utility API layouts."""

    errors: list[BaseException] = []
    for module_name in ("swift.utils", "swift.utils.torch_utils", "swift.llm"):
        try:
            module = __import__(module_name, fromlist=("to_device",))
            return getattr(module, "to_device")
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "ms-swift to_device is required for Stage2 collation; tried "
        "swift.utils.to_device, swift.utils.torch_utils.to_device, and swift.llm.to_device"
    ) from errors[-1]


def _require_trace_field(
    result: DetectionDecodeResult,
    field_name: str,
) -> list[Any]:
    value = getattr(result, field_name)
    if value is None:
        raise ValueError(
            f"decode trace requires {field_name} when trace_logprobs=true"
        )
    if not isinstance(value, list):
        raise ValueError(f"decode trace {field_name} must be a list")
    if result.text and len(value) == 0:
        raise ValueError(
            f"decode trace requires nonempty {field_name} for nonempty generated text"
        )
    return value


def validate_decode_trace(
    result: DetectionDecodeResult,
    *,
    trace_logprobs: bool,
) -> DetectionDecodeResult:
    if not trace_logprobs:
        return result

    token_ids = _require_trace_field(result, "generated_token_ids")
    tokens = _require_trace_field(result, "generated_tokens")
    logprobs = _require_trace_field(result, "generated_logprobs")

    lengths = {
        "generated_token_ids": len(token_ids),
        "generated_tokens": len(tokens),
        "generated_logprobs": len(logprobs),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"decode trace shape mismatch: {lengths}")

    for index, logprob in enumerate(logprobs):
        if not isinstance(logprob, (float, int)) or not math.isfinite(float(logprob)):
            raise ValueError(
                "decode trace generated_logprobs must contain finite values; "
                f"index {index} is {logprob!r}"
            )

    return result


def generate_batch(
    *,
    owner: Any,
    images: List[Image.Image],
    result_factory: Callable[..., Any],
) -> List[Any]:
    """Generate a micro-batch across supported infer backends."""

    if not images:
        return []

    backend = str(owner.cfg.backend_type).strip().lower()
    if backend == "hf":
        return generate_hf_batch(
            owner=owner,
            images=images,
            result_factory=result_factory,
        )
    if backend == "vllm":
        return generate_vllm_batch(
            owner=owner,
            images=images,
            result_factory=result_factory,
        )
    raise ValueError(f"infer.backend.type must be hf|vllm, got {backend!r}")


def generate_hf_batch(
    *,
    owner: Any,
    images: List[Image.Image],
    result_factory: Callable[..., Any],
) -> List[Any]:
    import torch

    from src.common.qwen_generation import (
        apply_qwen_chat_generation_token_ids,
        call_processor_with_qwen_geometry,
    )

    assert owner.model is not None and owner.processor is not None
    if not images:
        return []

    from src.infer.prompt import build_offline_detection_chat_messages

    messages = [
        build_offline_detection_chat_messages(
            system_prompt=getattr(owner, "system_prompt", ""),
            user_prompt=getattr(owner, "user_prompt", ""),
            image=img,
        )
        for img in images
    ]
    prompt_texts = [
        owner.processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        for message in messages
    ]

    model_inputs = call_processor_with_qwen_geometry(
        owner.processor,
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        key: value.to(owner.cfg.device) for key, value in model_inputs.items()
    }
    prompt_padded_len = int(model_inputs["input_ids"].shape[1])

    do_sample = owner.gen_cfg.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=owner.gen_cfg.max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(1e-4, owner.gen_cfg.temperature)
        gen_kwargs["top_p"] = owner.gen_cfg.top_p
    if owner.gen_cfg.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = owner.gen_cfg.repetition_penalty
    apply_qwen_chat_generation_token_ids(
        gen_kwargs,
        tokenizer=owner.processor.tokenizer,
    )

    trace_logprobs = bool(getattr(owner.gen_cfg, "trace_logprobs", False))

    with torch.inference_mode():
        try:
            gen_outputs = owner.model.generate(
                **model_inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
            )
        except (TypeError, ValueError) as exc:
            if trace_logprobs:
                raise RuntimeError(
                    "HF generated-token logprob tracing requires model.generate "
                    "support for return_dict_in_generate=True and output_scores=True"
                ) from exc
            owner.logger.warning(
                "HF trace capture unavailable; falling back to text-only generation: %s",
                exc,
            )
            gen_outputs = owner.model.generate(**model_inputs, **gen_kwargs)

    if isinstance(gen_outputs, torch.Tensor):
        gen_ids = gen_outputs
        scores = []
    else:
        gen_ids = gen_outputs.sequences
        scores = list(getattr(gen_outputs, "scores", ()) or ())
    gen_token_ids = gen_ids[:, prompt_padded_len:]
    generated_len = int(gen_token_ids.shape[1])
    if trace_logprobs and int(len(scores)) != generated_len:
        raise ValueError(
            "HF generated-token logprob trace shape mismatch: "
            f"generated_tokens={generated_len} scores={int(len(scores))}"
        )

    trace_len = generated_len if trace_logprobs else min(generated_len, int(len(scores)))
    token_logprobs_by_sample: List[List[float]] = [[] for _ in range(int(len(images)))]
    if trace_len > 0:
        for step_idx in range(trace_len):
            step_scores = scores[step_idx]
            if not isinstance(step_scores, torch.Tensor):
                continue
            step_token_ids = gen_token_ids[:, step_idx].long()
            step_scores_f = step_scores.float()
            step_selected_logits = step_scores_f.gather(
                dim=1, index=step_token_ids.unsqueeze(1)
            ).squeeze(1)
            step_log_norm = torch.logsumexp(step_scores_f, dim=-1)
            step_selected = step_selected_logits - step_log_norm
            selected_vals = step_selected.detach().cpu().tolist()
            for sample_idx, val in enumerate(selected_vals):
                token_logprobs_by_sample[sample_idx].append(float(val))

    out: List[Any] = []
    for idx in range(len(images)):
        gen_only = gen_token_ids[idx]
        raw_text = owner.processor.tokenizer.decode(
            gen_only,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        trace_token_ids = gen_token_ids[idx, :trace_len].detach().cpu().tolist()
        generated_token_text = (
            owner.processor.tokenizer.batch_decode(
                [[int(tok)] for tok in trace_token_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if trace_token_ids
            else []
        )
        token_logprobs = token_logprobs_by_sample[idx][: len(generated_token_text)]
        if trace_logprobs:
            validate_decode_trace(
                DetectionDecodeResult(
                    text=raw_text,
                    generated_token_ids=[int(tok) for tok in trace_token_ids],
                    generated_tokens=generated_token_text,
                    generated_logprobs=token_logprobs,
                    stop_reason=None,
                    backend="hf",
                    prompt_token_ids=[
                        int(tok)
                        for tok in model_inputs["input_ids"][idx].detach().cpu().tolist()
                    ],
                ),
                trace_logprobs=True,
            )

        out.append(
            result_factory(
                text=raw_text,
                generated_token_ids=[int(tok) for tok in trace_token_ids],
                generated_token_text=generated_token_text,
                token_logprobs=token_logprobs,
                prompt_token_ids=[
                    int(tok)
                    for tok in model_inputs["input_ids"][idx].detach().cpu().tolist()
                ],
                stop_reason=None,
                error=None,
            )
        )
    return out


def generate_vllm_batch(
    *,
    owner: Any,
    images: List[Image.Image],
    result_factory: Callable[..., Any],
) -> List[Any]:
    if not images:
        return []

    if vllm_backend_mode(owner) == "local":
        results = generate_vllm_local_batch(
            owner=owner,
            images=images,
            result_factory=result_factory,
        )
        if bool(getattr(owner.gen_cfg, "trace_logprobs", False)):
            for result in results:
                validate_decode_trace(
                    DetectionDecodeResult(
                        text=str(getattr(result, "text", "") or ""),
                        generated_token_ids=getattr(result, "generated_token_ids", None),
                        generated_tokens=getattr(result, "generated_token_text", None),
                        generated_logprobs=getattr(result, "token_logprobs", None),
                        stop_reason=getattr(result, "stop_reason", None),
                        backend="vllm",
                    ),
                    trace_logprobs=True,
                )
        return results

    max_workers_raw = owner.cfg.backend.get("client_concurrency")
    try:
        max_workers = (
            int(max_workers_raw)
            if max_workers_raw is not None
            else int(len(images))
        )
    except (TypeError, ValueError):
        max_workers = int(len(images))
    max_workers = max(1, min(int(max_workers), int(len(images))))

    results: List[Any | None] = [None for _ in images]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_to_idx = {
            executor.submit(
                generate_vllm_server_result,
                owner=owner,
                image=image,
                result_factory=result_factory,
            ): idx
            for idx, image in enumerate(images)
        }
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()

    if any(result is None for result in results):
        raise RuntimeError("vLLM generation returned missing outputs")
    return [result for result in results if result is not None]


def vllm_backend_mode(owner: Any) -> str:
    mode_raw = (owner.cfg.backend or {}).get("mode", "server")
    mode = str(mode_raw or "server").strip().lower()
    if mode not in {"local", "server"}:
        raise ValueError(
            "infer.backend.mode must be one of {'local', 'server'} "
            "for infer.backend.type=vllm"
        )
    return mode


def load_vllm_local_backend(owner: Any) -> None:
    if getattr(owner, "vllm_llm", None) is not None:
        return

    import os
    from pathlib import Path

    # vLLM's CLI defaults to "spawn" for safety, but the library API does not.
    # Without this, using vLLM inside a process that already touched CUDA can fail with:
    #   "Cannot re-initialize CUDA in forked subprocess"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM local backend requires the 'vllm' package. Install it in the ms env, or set infer.backend.type=hf."
        ) from exc

    from src.common.qwen_generation import qwen_processor_call_kwargs

    model = str(owner.cfg.backend.get("model") or owner.cfg.model_checkpoint).strip()
    if not model:
        raise RuntimeError(
            "infer.backend.model (or infer.model_checkpoint) is required for vLLM local mode"
        )

    # Reuse server_options-style knobs when present for reproducibility.
    server_opts = owner.cfg.backend.get("server_options") or {}
    allowed_local_media_path = str(
        owner.cfg.root_image_dir
        or os.environ.get("ROOT_IMAGE_DIR")
        or Path(owner.cfg.gt_jsonl).parent.resolve()
    )

    kwargs: dict[str, Any] = {}

    tp = server_opts.get("vllm_tensor_parallel_size", None)
    if tp is not None:
        try:
            kwargs["tensor_parallel_size"] = int(tp)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "infer.backend.server_options.vllm_tensor_parallel_size must be int-compatible"
            ) from exc

    util = server_opts.get("vllm_gpu_memory_utilization", None)
    if util is not None:
        try:
            kwargs["gpu_memory_utilization"] = float(util)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "infer.backend.server_options.vllm_gpu_memory_utilization must be float-compatible"
            ) from exc

    # `max_model_len` is a vLLM kwarg (mirrors --max-model-len on the server).
    mml = server_opts.get("vllm_max_model_len", None)
    if mml is not None:
        try:
            kwargs["max_model_len"] = int(mml)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "infer.backend.server_options.vllm_max_model_len must be int-compatible"
            ) from exc

    owner.vllm_llm = LLM(
        model=model,
        trust_remote_code=True,
        allowed_local_media_path=str(allowed_local_media_path or ""),
        seed=int(owner.gen_cfg.seed) if owner.gen_cfg.seed is not None else None,
        mm_processor_kwargs=qwen_processor_call_kwargs(),
        **kwargs,
    )


def vllm_sampling_params(owner: Any) -> Any:
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requires the 'vllm' package. Install it in the ms env, or set infer.backend.type=hf."
        ) from exc

    from src.common.detection_sequence import IM_END_TOKEN

    kwargs: dict[str, Any] = {
        "temperature": float(owner.gen_cfg.temperature),
        "top_p": float(owner.gen_cfg.top_p),
        "max_tokens": int(owner.gen_cfg.max_new_tokens),
        "repetition_penalty": float(owner.gen_cfg.repetition_penalty or 1.0),
        "seed": int(owner.gen_cfg.seed) if owner.gen_cfg.seed is not None else None,
        "stop": [IM_END_TOKEN],
    }
    if bool(getattr(owner.gen_cfg, "trace_logprobs", False)):
        kwargs["logprobs"] = 1
    return SamplingParams(**kwargs)


def validate_vllm_server_backend(owner: Any) -> None:
    """Fail fast on global vLLM server misconfiguration/unavailability."""

    if vllm_backend_mode(owner) == "local":
        return

    import os

    base_url = str(
        owner.cfg.backend.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
    ).strip()
    if not base_url:
        raise RuntimeError(
            "infer.backend.type=vllm requires infer.backend.base_url (or env VLLM_BASE_URL) when backend.mode=server. "
            "To run without a server, set infer.backend.mode=local. To disable vLLM, set infer.backend.type=hf."
        )

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requires the 'requests' package. Install it in the ms env, or set infer.backend.type=hf."
        ) from exc

    timeout_s = float(owner.cfg.backend.get("timeout_s", 3.0))
    root = base_url.rstrip("/")
    models_url = (root + "/models") if root.endswith("/v1") else (root + "/v1/models")
    try:
        resp = requests.get(models_url, timeout=timeout_s)
    except (requests.exceptions.RequestException, OSError, ValueError) as exc:
        raise RuntimeError(
            "Failed to reach vLLM server for infer.backend.type=vllm. "
            f"Tried GET {models_url}. To disable vLLM, set infer.backend.type=hf."
        ) from exc
    if int(getattr(resp, "status_code", 0) or 0) >= 400:
        raise RuntimeError(
            "vLLM server preflight check failed for infer.backend.type=vllm. "
            f"GET {models_url} returned status={resp.status_code}. To disable vLLM, set infer.backend.type=hf."
        )


def generate_vllm_local_batch(
    *,
    owner: Any,
    images: list[Any],
    result_factory: Callable[..., Any],
) -> list[Any]:
    """Generate via the in-process vLLM Python API (no HTTP server)."""

    if not images:
        return []

    load_vllm_local_backend(owner)
    assert owner.vllm_llm is not None

    try:
        import base64
        import io
    except ImportError as exc:
        return [result_factory(text="", error=exc) for _ in images]

    from src.infer.prompt import build_offline_detection_chat_messages

    msg_batch = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        msg_batch.append(
            build_offline_detection_chat_messages(
                system_prompt=owner.system_prompt,
                user_prompt=owner.user_prompt,
                image={"url": f"data:image/png;base64,{b64}"},
                image_content_type="image_url",
            )
        )

    sp = vllm_sampling_params(owner)
    outs = None
    chat_exc: Exception | None = None
    try:
        outs = owner.vllm_llm.chat(msg_batch, sampling_params=sp, use_tqdm=False)
    except Exception as exc:  # noqa: BLE001
        chat_exc = exc
        outs = None

    if outs is None:
        if chat_exc is None:
            chat_exc = RuntimeError("vLLM local chat() produced no outputs")
        return [result_factory(text="", error=chat_exc) for _ in images]

    trace_logprobs = bool(getattr(owner.gen_cfg, "trace_logprobs", False))
    tokenizer = getattr(getattr(owner, "processor", None), "tokenizer", None)

    results: list[Any] = []
    for out in outs:
        seqs = getattr(out, "outputs", None) or []
        seq = seqs[0] if seqs else None
        text = str(getattr(seq, "text", "") or "") if seq is not None else ""
        generated_token_ids = None
        generated_token_text = None
        token_logprobs = None
        prompt_token_ids = None
        stop_reason = None
        if trace_logprobs:
            generated_token_ids = _coerce_int_list(getattr(seq, "token_ids", None))
            if generated_token_ids is None:
                raise ValueError(
                    "vLLM local trace requires generated token_ids in RequestOutput.outputs[0]"
                )
            token_logprobs, generated_token_text = extract_vllm_local_logprobs(
                getattr(seq, "logprobs", None),
                token_ids=generated_token_ids,
                tokenizer=tokenizer,
            )
            prompt_token_ids = _coerce_int_list(getattr(out, "prompt_token_ids", None))
            stop_reason = getattr(seq, "finish_reason", None) or getattr(
                seq, "stop_reason", None
            )
            validate_decode_trace(
                DetectionDecodeResult(
                    text=text,
                    generated_token_ids=generated_token_ids,
                    generated_tokens=generated_token_text,
                    generated_logprobs=token_logprobs,
                    stop_reason=stop_reason,
                    backend="vllm",
                    prompt_token_ids=prompt_token_ids,
                ),
                trace_logprobs=True,
            )
        results.append(
            result_factory(
                text=text,
                generated_token_ids=generated_token_ids,
                generated_token_text=generated_token_text,
                token_logprobs=token_logprobs,
                prompt_token_ids=prompt_token_ids,
                stop_reason=stop_reason,
                error=None,
            )
        )

    if len(results) != len(images):
        raise RuntimeError(
            f"vLLM returned {len(results)} outputs for {len(images)} requests"
        )

    return results


def generate_vllm_server_result(
    *,
    owner: Any,
    image: Any,
    result_factory: Callable[..., Any],
) -> Any:
    """Generate via an OpenAI-compatible vLLM server and preserve trace fields."""

    try:
        import base64
        import io
        import os

        import requests
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requires the 'requests' package. Install it in the ms env, or set infer.backend.type=hf."
        ) from exc

    from src.common.detection_sequence import IM_END_TOKEN
    from src.infer.prompt import build_offline_detection_chat_messages

    base_url = str(
        owner.cfg.backend.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
    ).strip()
    if not base_url:
        raise RuntimeError(
            "infer.backend.type=vllm requires infer.backend.base_url (or env VLLM_BASE_URL). "
            "To run without a server, set infer.backend.mode=local. To disable vLLM, set infer.backend.type=hf."
        )

    model = str(owner.cfg.backend.get("model") or owner.cfg.model_checkpoint).strip()
    timeout_s = float(owner.cfg.backend.get("timeout_s", 180.0))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    messages = build_offline_detection_chat_messages(
        system_prompt=owner.system_prompt,
        user_prompt=owner.user_prompt,
        image={"url": f"data:image/png;base64,{b64}"},
        image_content_type="image_url",
    )

    base_url = base_url.rstrip("/")
    url = (
        base_url + "/chat/completions"
        if base_url.endswith("/v1")
        else base_url + "/v1/chat/completions"
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(owner.gen_cfg.temperature),
        "top_p": float(owner.gen_cfg.top_p),
        "max_tokens": int(owner.gen_cfg.max_new_tokens),
        "stream": False,
        "stop": [IM_END_TOKEN],
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if owner.gen_cfg.repetition_penalty is not None:
        payload["repetition_penalty"] = float(owner.gen_cfg.repetition_penalty)
    if owner.gen_cfg.seed is not None:
        payload["seed"] = int(owner.gen_cfg.seed)
    trace_logprobs = bool(getattr(owner.gen_cfg, "trace_logprobs", False))
    if trace_logprobs:
        payload["logprobs"] = True
        payload["return_token_ids"] = True
        payload["return_tokens_as_token_ids"] = True

    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"vLLM server error status={resp.status_code}: {resp.text[:2000]}"
        )

    result = normalize_vllm_trace_response(
        resp.json(),
        trace_logprobs=trace_logprobs,
        backend_mode="openai-compatible",
    )
    return result_factory(
        text=result.text,
        generated_token_ids=result.generated_token_ids,
        generated_token_text=result.generated_tokens,
        token_logprobs=result.generated_logprobs,
        prompt_token_ids=result.prompt_token_ids,
        stop_reason=result.stop_reason,
        error=None,
    )


def _get_first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _coerce_int_list(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    return [int(item) for item in value]


def _coerce_float_list(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]


def _coerce_str_list(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _extract_openai_choice_text(choice: Mapping[str, Any]) -> str:
    raw_text = choice.get("text")
    if isinstance(raw_text, str):
        return raw_text
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "") or ""))
        return "".join(parts)
    return ""


def extract_swift_choice_logprobs(logprobs_raw: Any) -> tuple[list[float], list[str]]:
    """Extract ms-swift/vLLM per-token logprobs without clipping or padding."""

    if not isinstance(logprobs_raw, Mapping):
        raise RuntimeError(
            "Missing vLLM logprobs in server response; ensure request_config.logprobs=true"
        )
    content = logprobs_raw.get("content")
    if not isinstance(content, list):
        raise RuntimeError(
            "Malformed vLLM logprobs payload: expected logprobs.content as a list"
        )

    token_logprobs: list[float] = []
    generated_token_text: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise RuntimeError(
                "Malformed vLLM logprobs payload: expected items in logprobs.content to be mappings"
            )
        generated_token_text.append(str(item.get("token") or ""))
        lp_raw = item.get("logprob")
        try:
            lp = float(lp_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed vLLM logprobs payload: non-numeric logprob={lp_raw!r}"
            ) from exc
        if not math.isfinite(lp):
            raise RuntimeError(
                f"Malformed vLLM logprobs payload: non-finite logprob={lp_raw!r}"
            )
        token_logprobs.append(lp)

    return token_logprobs, generated_token_text


def _logprob_payload_value(payload: Any) -> tuple[float, str | None]:
    if isinstance(payload, Mapping):
        lp_raw = payload.get("logprob")
        token_raw = payload.get("decoded_token", payload.get("token"))
    else:
        lp_raw = getattr(payload, "logprob", payload)
        token_raw = getattr(payload, "decoded_token", None)
        if token_raw is None:
            token_raw = getattr(payload, "token", None)
    try:
        lp = float(lp_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed vLLM local logprobs payload: non-numeric logprob={lp_raw!r}"
        ) from exc
    if not math.isfinite(lp):
        raise RuntimeError(
            f"Malformed vLLM local logprobs payload: non-finite logprob={lp_raw!r}"
        )
    token_text = str(token_raw) if token_raw is not None else None
    return lp, token_text


def extract_vllm_local_logprobs(
    logprobs_raw: Any,
    *,
    token_ids: list[int],
    tokenizer: Any | None = None,
) -> tuple[list[float], list[str]]:
    """Extract vLLM RequestOutput per-token logprobs without clipping/padding."""

    if not isinstance(logprobs_raw, list):
        raise RuntimeError(
            "Missing vLLM local logprobs; ensure SamplingParams.logprobs is set for trace_logprobs=true"
        )
    if len(logprobs_raw) != len(token_ids):
        raise RuntimeError(
            "vLLM local logprobs length mismatch: "
            f"logprobs={len(logprobs_raw)} token_ids={len(token_ids)}"
        )

    token_logprobs: list[float] = []
    generated_token_text: list[str] = []
    tokenizer_pieces = (
        decode_token_pieces_with_tokenizer(tokenizer=tokenizer, token_ids=token_ids)
        if tokenizer is not None
        else None
    )

    for index, (token_id, item) in enumerate(zip(token_ids, logprobs_raw)):
        selected = item
        if isinstance(item, Mapping):
            selected = item.get(int(token_id))
            if selected is None:
                selected = item.get(str(int(token_id)))
            if selected is None:
                raise RuntimeError(
                    "Malformed vLLM local logprobs payload: missing chosen token "
                    f"logprob for token_id={int(token_id)}"
                )
        lp, token_text = _logprob_payload_value(selected)
        if tokenizer_pieces is not None:
            token_text = tokenizer_pieces[index]
        if token_text is None:
            raise RuntimeError(
                "Malformed vLLM local logprobs payload: missing decoded token text "
                f"for token_id={int(token_id)}"
            )
        token_logprobs.append(lp)
        generated_token_text.append(token_text)

    return token_logprobs, generated_token_text


def strip_left_padding_token_ids(
    token_ids: Optional[list[int]],
    *,
    pad_token_id: Any | None,
) -> Optional[list[int]]:
    if token_ids is None:
        return None
    ids = [int(t) for t in token_ids]
    try:
        pad = int(pad_token_id) if pad_token_id is not None else None
    except (TypeError, ValueError):
        pad = None
    if pad is None:
        return ids

    i = 0
    while i < int(len(ids)) and int(ids[i]) == int(pad):
        i += 1
    if i == 0:
        return ids
    trimmed = ids[i:]
    return trimmed if trimmed else ids


_strip_left_padding_token_ids = strip_left_padding_token_ids


def decode_token_pieces_with_tokenizer(
    *,
    tokenizer: Any,
    token_ids: list[int],
) -> list[str]:
    out: list[str] = []
    for token_id in token_ids:
        try:
            piece = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            piece = tokenizer.decode([int(token_id)])
        out.append(str(piece))
    return out


def normalize_vllm_trace_response(
    response: dict[str, Any],
    *,
    trace_logprobs: bool,
    backend_mode: str,
    pad_token_id: Any | None = None,
    tokenizer: Any | None = None,
) -> DetectionDecodeResult:
    if isinstance(response, dict):
        wrapped = response.get("response")
        if isinstance(wrapped, dict):
            response = wrapped
        if response.get("object") == "error":
            raise RuntimeError(str(response.get("message") or response))

    if "details" in response:
        details = response["details"] or {}
        text = str(response.get("text") or details.get("text") or "")
        prompt_token_ids = _strip_left_padding_token_ids(
            _coerce_int_list(details.get("prompt_token_ids")),
            pad_token_id=(
                getattr(tokenizer, "pad_token_id", None)
                if tokenizer is not None
                else pad_token_id
            ),
        )
        generated_token_ids = _coerce_int_list(
            _get_first_present(details, ("generated_token_ids", "token_ids"))
        )
        generated_tokens = _coerce_str_list(
            _get_first_present(details, ("generated_tokens", "tokens"))
        )
        if tokenizer is not None and generated_token_ids is not None:
            generated_tokens = decode_token_pieces_with_tokenizer(
                tokenizer=tokenizer,
                token_ids=generated_token_ids,
            )
        if trace_logprobs and text and not prompt_token_ids:
            raise ValueError(
                "vLLM ms-swift trace requires nonempty prompt_token_ids "
                "when trace_logprobs=true"
            )
        result = DetectionDecodeResult(
            text=text,
            generated_token_ids=generated_token_ids,
            generated_tokens=generated_tokens,
            generated_logprobs=_coerce_float_list(
                _get_first_present(details, ("generated_logprobs", "logprobs"))
            ),
            stop_reason=response.get("stop_reason") or details.get("stop_reason"),
            backend="vllm",
            backend_metadata={"backend_mode": backend_mode, "response_family": "ms-swift"},
            prompt_token_ids=prompt_token_ids,
        )
        return validate_decode_trace(result, trace_logprobs=trace_logprobs)

    if "choices" in response:
        choices = response.get("choices") or []
        if not choices:
            raise ValueError(
                f"vLLM {backend_mode} response requires at least one choice"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise ValueError(
                f"vLLM {backend_mode} response choices[0] must be a mapping"
            )
        text = _extract_openai_choice_text(first_choice)
        if str(backend_mode).strip().lower() == "ms-swift":
            prompt_token_ids = _strip_left_padding_token_ids(
                _coerce_int_list(response.get("prompt_token_ids")),
                pad_token_id=(
                    getattr(tokenizer, "pad_token_id", None)
                    if tokenizer is not None
                    else pad_token_id
                ),
            )
            if not prompt_token_ids:
                raise ValueError(
                    "vLLM ms-swift response requires nonempty prompt_token_ids "
                    "when return_details=true"
                )
            generated_token_ids = _coerce_int_list(first_choice.get("token_ids"))
            if generated_token_ids is None:
                raise ValueError(
                    "vLLM ms-swift response requires token_ids "
                    "when return_details=true"
                )
            generated_tokens = None
            generated_logprobs = None
            if trace_logprobs:
                generated_logprobs, generated_tokens = extract_swift_choice_logprobs(
                    first_choice.get("logprobs")
                )
                if tokenizer is not None:
                    generated_tokens = decode_token_pieces_with_tokenizer(
                        tokenizer=tokenizer,
                        token_ids=generated_token_ids,
                    )
            result = DetectionDecodeResult(
                text=text,
                generated_token_ids=generated_token_ids,
                generated_tokens=generated_tokens,
                generated_logprobs=generated_logprobs,
                stop_reason=first_choice.get("finish_reason"),
                backend="vllm",
                backend_metadata={
                    "backend_mode": backend_mode,
                    "response_family": "ms-swift",
                },
                prompt_token_ids=prompt_token_ids,
            )
            return validate_decode_trace(result, trace_logprobs=trace_logprobs)

        result = DetectionDecodeResult(
            text=text,
            generated_token_ids=_coerce_int_list(
                _get_first_present(first_choice, ("token_ids", "generated_token_ids"))
            )
            if trace_logprobs
            else None,
            generated_tokens=None,
            generated_logprobs=None,
            stop_reason=first_choice.get("finish_reason"),
            backend="vllm",
            backend_metadata={
                "backend_mode": backend_mode,
                "response_family": "openai-compatible",
            },
            prompt_token_ids=(
                _strip_left_padding_token_ids(
                    _coerce_int_list(response.get("prompt_token_ids")),
                    pad_token_id=(
                        getattr(tokenizer, "pad_token_id", None)
                        if tokenizer is not None
                        else pad_token_id
                    ),
                )
                if trace_logprobs
                else None
            ),
        )
        if trace_logprobs:
            generated_logprobs, generated_tokens = extract_swift_choice_logprobs(
                first_choice.get("logprobs")
            )
            generated_token_ids = result.generated_token_ids
            if generated_token_ids is None:
                generated_token_ids = _coerce_int_list(response.get("token_ids"))
            if tokenizer is not None and generated_token_ids is not None:
                generated_tokens = decode_token_pieces_with_tokenizer(
                    tokenizer=tokenizer,
                    token_ids=generated_token_ids,
                )
            result = DetectionDecodeResult(
                text=text,
                generated_token_ids=generated_token_ids,
                generated_tokens=generated_tokens,
                generated_logprobs=generated_logprobs,
                stop_reason=first_choice.get("finish_reason"),
                backend="vllm",
                backend_metadata={
                    "backend_mode": backend_mode,
                    "response_family": "openai-compatible",
                },
                prompt_token_ids=result.prompt_token_ids,
            )
        return validate_decode_trace(result, trace_logprobs=trace_logprobs)

    raise ValueError("unsupported vLLM response shape")
