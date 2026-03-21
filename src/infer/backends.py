from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List

import torch
from PIL import Image


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
    assert owner.model is not None and owner.processor is not None
    if not images:
        return []

    messages = [owner._build_messages(img) for img in images]
    prompt_texts = [
        owner.processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        for message in messages
    ]

    model_inputs = owner.processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        key: value.to(owner.cfg.device) for key, value in model_inputs.items()
    }

    gen_kwargs = dict(
        max_new_tokens=owner.gen_cfg.max_new_tokens,
        do_sample=owner.gen_cfg.temperature > 0,
        temperature=max(1e-4, owner.gen_cfg.temperature),
        top_p=owner.gen_cfg.top_p,
        use_cache=True,
    )
    if owner.gen_cfg.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = owner.gen_cfg.repetition_penalty

    with torch.inference_mode():
        try:
            gen_outputs = owner.model.generate(
                **model_inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
            )
        except (TypeError, ValueError) as exc:
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
    prompt_padded_len = int(model_inputs["input_ids"].shape[1])
    gen_token_ids = gen_ids[:, prompt_padded_len:]

    trace_len = min(int(gen_token_ids.shape[1]), int(len(scores)))
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

        out.append(
            result_factory(
                text=raw_text,
                generated_token_text=generated_token_text,
                token_logprobs=token_logprobs,
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

    if owner._vllm_mode() == "local":
        return owner._generate_vllm_local_batch(images)

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

    texts: List[str | None] = [None for _ in images]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_to_idx = {
            executor.submit(owner._generate_vllm_server, image): idx
            for idx, image in enumerate(images)
        }
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            texts[idx] = fut.result()

    if any(text is None for text in texts):
        raise RuntimeError("vLLM generation returned missing outputs")

    return [result_factory(text=str(text), error=None) for text in texts]
