import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from src.coord_tokens.codec import get_coord_token_ids


def _resolve_text_vocab_size(model: Any) -> int | None:
    config = getattr(model, "config", None)
    text_cfg = getattr(config, "text_config", None)
    text_vocab = getattr(text_cfg, "vocab_size", None)
    if isinstance(text_vocab, int) and text_vocab > 0:
        return int(text_vocab)
    vocab = getattr(config, "vocab_size", None)
    if isinstance(vocab, int) and vocab > 0:
        return int(vocab)
    return None

def _resolve_embedding_rows(model: Any) -> int | None:
    get_emb = getattr(model, "get_input_embeddings", None)
    if not callable(get_emb):
        return None
    try:
        embeddings = get_emb()
    except Exception:
        return None
    weight = getattr(embeddings, "weight", None)
    shape = getattr(weight, "shape", None)
    if not shape or len(shape) < 1:
        return None
    try:
        rows = int(shape[0])
    except (TypeError, ValueError):
        return None
    return rows if rows > 0 else None

def _validate_batch_contract(
    *,
    model: Any,
    inputs: Mapping[str, Any],
    template: Any = None,
) -> None:
    input_ids = inputs.get("input_ids")
    flat_input_ids: torch.Tensor | None = None
    batch_size: int | None = None
    seq_len: int | None = None
    if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
        embed_rows = _resolve_embedding_rows(model)
        if embed_rows is not None:
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= embed_rows:
                raise ValueError(
                    "Batch input_ids violate embedding range before model forward: "
                    f"min_id={min_id} max_id={max_id} embed_rows={embed_rows} "
                    f"shape={tuple(input_ids.shape)}"
                )
        flat_input_ids = input_ids.reshape(-1)
        if input_ids.ndim == 1:
            batch_size = 1
            seq_len = int(input_ids.shape[0])
        elif input_ids.ndim >= 2:
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[-1])

    labels = inputs.get("labels")
    if isinstance(labels, torch.Tensor) and labels.numel() > 0:
        vocab_size = _resolve_text_vocab_size(model)
        if vocab_size is not None:
            invalid = (labels < -100) | ((labels != -100) & (labels >= vocab_size))
            if bool(invalid.any().item()):
                bad_values = torch.unique(labels[invalid]).detach().cpu().tolist()
                bad_values = [int(v) for v in bad_values[:16]]
                raise ValueError(
                    "Batch labels violate causal-LM target range before model forward: "
                    f"vocab_size={vocab_size} bad_values={bad_values} "
                    f"shape={tuple(labels.shape)}"
                )

    image_grid = inputs.get("image_grid_thw")
    pixel_values = inputs.get("pixel_values")
    image_token_id = getattr(template, "image_token_id", None)
    if isinstance(image_grid, torch.Tensor) and image_grid.numel() > 0:
        if image_grid.ndim != 2 or int(image_grid.shape[-1]) != 3:
            raise ValueError(
                "Batch image_grid_thw must have shape [num_images, 3] before model forward: "
                f"shape={tuple(image_grid.shape)}"
            )
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 1:
            expected_visual_rows = int(image_grid.prod(dim=-1).sum().item())
            actual_visual_rows = int(pixel_values.shape[0])
            if actual_visual_rows != expected_visual_rows:
                raise ValueError(
                    "Batch pixel_values/image_grid_thw visual-row mismatch before model forward: "
                    f"expected_rows={expected_visual_rows} actual_rows={actual_visual_rows} "
                    f"pixel_values.shape={tuple(pixel_values.shape)} image_grid_thw.shape={tuple(image_grid.shape)}"
                )
        if flat_input_ids is not None and isinstance(image_token_id, int):
            processor = getattr(template, "processor", None)
            image_processor = getattr(processor, "image_processor", None)
            merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
            if merge_size <= 0:
                merge_size = 1
            expected = int((image_grid.prod(dim=-1) // (merge_size**2)).sum().item())
            actual = int((flat_input_ids == int(image_token_id)).sum().item())
            if actual != expected:
                preview = image_grid[:8].detach().cpu().tolist()
                raise ValueError(
                    "Batch image token count mismatches image_grid_thw before model forward: "
                    f"expected={expected} actual={actual} merge_size={merge_size} "
                    f"image_grid_preview={preview} input_shape={tuple(input_ids.shape)}"
                )

    if batch_size is None or seq_len is None:
        return

    position_ids = inputs.get("position_ids")
    if isinstance(position_ids, torch.Tensor):
        if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3:
            raise ValueError(
                "Batch position_ids must have shape [3, batch, seq] before model forward: "
                f"shape={tuple(position_ids.shape)}"
            )
        if int(position_ids.shape[1]) != batch_size or int(position_ids.shape[2]) != seq_len:
            raise ValueError(
                "Batch position_ids batch/seq dims mismatch input_ids before model forward: "
                f"position_ids.shape={tuple(position_ids.shape)} input_shape={tuple(input_ids.shape)}"
            )

    text_position_ids = inputs.get("text_position_ids")
    if isinstance(text_position_ids, torch.Tensor):
        if text_position_ids.ndim != 2:
            raise ValueError(
                "Batch text_position_ids must have shape [batch, seq] before model forward: "
                f"shape={tuple(text_position_ids.shape)}"
            )
        if int(text_position_ids.shape[0]) != batch_size or int(text_position_ids.shape[1]) != seq_len:
            raise ValueError(
                "Batch text_position_ids batch/seq dims mismatch input_ids before model forward: "
                f"text_position_ids.shape={tuple(text_position_ids.shape)} input_shape={tuple(input_ids.shape)}"
            )

    cu_q = inputs.get("cu_seq_lens_q")
    cu_k = inputs.get("cu_seq_lens_k")
    if isinstance(cu_q, torch.Tensor):
        if cu_q.ndim != 1:
            raise ValueError(
                "Batch cu_seq_lens_q must be a 1D tensor before model forward: "
                f"shape={tuple(cu_q.shape)}"
            )
        cu_list = [int(x) for x in cu_q.detach().cpu().tolist()]
        if not cu_list or cu_list[0] != 0 or cu_list[-1] != seq_len:
            raise ValueError(
                "Batch cu_seq_lens_q boundaries mismatch input_ids before model forward: "
                f"cu_seq_lens_q={cu_list} seq_len={seq_len}"
            )
        if any(b <= a for a, b in zip(cu_list, cu_list[1:])):
            raise ValueError(
                f"Batch cu_seq_lens_q must be strictly increasing before model forward: {cu_list}"
            )
        if isinstance(cu_k, torch.Tensor):
            cu_k_list = [int(x) for x in cu_k.detach().cpu().tolist()]
            if cu_k_list != cu_list:
                raise ValueError(
                    "Batch cu_seq_lens_k mismatch cu_seq_lens_q before model forward: "
                    f"cu_seq_lens_q={cu_list} cu_seq_lens_k={cu_k_list}"
                )
        seg_lens = [b - a for a, b in zip(cu_list, cu_list[1:])]
        max_seg_len = max(seg_lens) if seg_lens else 0
        max_length_q = inputs.get("max_length_q")
        max_length_k = inputs.get("max_length_k")
        if max_length_q is not None and int(max_length_q) != int(max_seg_len):
            raise ValueError(
                "Batch max_length_q mismatch cu_seq_lens_q before model forward: "
                f"max_length_q={int(max_length_q)} max_seg_len={int(max_seg_len)} cu_seq_lens_q={cu_list}"
            )
        if max_length_k is not None and int(max_length_k) != int(max_seg_len):
            raise ValueError(
                "Batch max_length_k mismatch cu_seq_lens_k before model forward: "
                f"max_length_k={int(max_length_k)} max_seg_len={int(max_seg_len)} cu_seq_lens_q={cu_list}"
            )
        if isinstance(text_position_ids, torch.Tensor) and batch_size == 1:
            starts = torch.nonzero(text_position_ids[0] == 0, as_tuple=False).flatten().detach().cpu().tolist()
            starts = [int(x) for x in starts]
            if starts != cu_list[:-1]:
                raise ValueError(
                    "Batch text_position_ids reset points mismatch cu_seq_lens_q before model forward: "
                    f"starts={starts[:32]} cu_seq_lens_q={cu_list[:32]}"
                )
        pack_num_samples = inputs.get("pack_num_samples")
        if isinstance(pack_num_samples, torch.Tensor) and pack_num_samples.numel() == batch_size and batch_size == 1:
            expected_segments = int(pack_num_samples.reshape(-1)[0].item())
            if expected_segments > 0 and int(len(cu_list) - 1) != expected_segments:
                raise ValueError(
                    "Batch pack_num_samples mismatch cu_seq_lens_q segment count before model forward: "
                    f"pack_num_samples={expected_segments} segment_count={int(len(cu_list)-1)} cu_seq_lens_q={cu_list}"
                )

class GradAccumLossScaleMixin:
    """CoordExp Trainer compatibility shim for packing-aware metrics.

    Background
    - We attach additional logging/diagnostic behavior without editing upstream model files.
    - For transformers>=4.57 + ms-swift, gradient-accumulation scaling is handled in the
      underlying ms-swift trainer (via the `num_items_in_batch` plumbing). We intentionally
      do NOT rescale the returned loss here to avoid double-scaling.

    Responsibilities
    - Pop packing metadata (`pack_num_samples`) and stash it on `self` for later metrics.
    - Log a few optimizer/runtime scalars as metrics so they appear in both train/eval logs.
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        # Pop known batch-extras once (Stage-1 / standard SFT). They are diagnostics-only
        # fields and MUST NOT be forwarded into model(**inputs).
        from src.trainers.batch_extras import pop_and_stash_batch_extras

        input_keys_before = (
            sorted(str(key) for key in inputs.keys()) if isinstance(inputs, Mapping) else []
        )
        extras = pop_and_stash_batch_extras(self, inputs)
        _maybe_dump_missing_prefix_denoising_sidecar(
            trainer=self,
            model=model,
            input_keys_before=input_keys_before,
            input_keys_after=(
                sorted(str(key) for key in inputs.keys())
                if isinstance(inputs, Mapping)
                else []
            ),
            extras=extras,
        )

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Log a few optimizer/runtime scalars as *metrics* so eval logs include them
        # (ms-swift only injects learning_rate/grad_norm into train logs by default).
        from src.metrics.reporter import SwiftMetricReporter, best_effort

        reporter = SwiftMetricReporter(self)

        def _log_runtime_metrics() -> None:
            lr_fn = getattr(self, "_get_learning_rate", None)
            if callable(lr_fn):
                reporter.update("learning_rate", float(lr_fn()))

            args = getattr(self, "args", None)
            gas = getattr(args, "gradient_accumulation_steps", None)
            if gas is not None:
                reporter.update("accum/grad_steps", float(gas))

            cur_gas = getattr(self, "current_gradient_accumulation_steps", None)
            if cur_gas is not None:
                reporter.update("accum/current_grad_steps", float(cur_gas))

        best_effort(self, name="runtime_metrics", fn=_log_runtime_metrics)

        return (loss, outputs) if return_outputs else loss


def _maybe_dump_missing_prefix_denoising_sidecar(
    *,
    trainer: Any,
    model: Any,
    input_keys_before: list[str],
    input_keys_after: list[str],
    extras: Any,
) -> None:
    prefix_denoising_cfg = getattr(trainer, "prefix_denoising_cfg", None)
    if not bool(prefix_denoising_cfg and getattr(prefix_denoising_cfg, "enabled", False)):
        return
    if getattr(extras, "prefix_denoising_hybrid", None) is not None:
        return
    args = getattr(trainer, "args", None)
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return
    rank = str(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or "0")
    path = Path(str(output_dir)) / f"prefix_denoising_missing_sidecar.rank{rank}.json"
    if path.exists():
        return
    try:
        data_collator = getattr(trainer, "data_collator", None)
        injected_collator = getattr(trainer, "_coordexp_injected_data_collator", None)
        state = getattr(trainer, "state", None)
        eval_dataset = getattr(trainer, "eval_dataset", None)
        train_dataset = getattr(trainer, "train_dataset", None)
        payload = {
            "rank": rank,
            "model_training": bool(getattr(model, "training", False)),
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "input_keys_before": input_keys_before,
            "input_keys_after": input_keys_after,
            "data_collator_type": (
                type(data_collator).__qualname__ if data_collator is not None else None
            ),
            "data_collator_module": (
                type(data_collator).__module__ if data_collator is not None else None
            ),
            "injected_collator_type": (
                type(injected_collator).__qualname__
                if injected_collator is not None
                else None
            ),
            "injected_collator_module": (
                type(injected_collator).__module__
                if injected_collator is not None
                else None
            ),
            "remove_unused_columns": bool(getattr(args, "remove_unused_columns", False)),
            "eval_strategy": str(getattr(args, "eval_strategy", "")),
            "eval_steps": getattr(args, "eval_steps", None),
            "save_strategy": str(getattr(args, "save_strategy", "")),
            "save_steps": getattr(args, "save_steps", None),
            "train_dataset": _dataset_debug_summary(train_dataset),
            "eval_dataset": _dataset_debug_summary(eval_dataset),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _dataset_debug_summary(dataset: Any) -> dict[str, Any]:
    if dataset is None:
        return {"type": None}
    payload: dict[str, Any] = {
        "type": type(dataset).__qualname__,
        "module": type(dataset).__module__,
    }
    try:
        payload["len"] = int(len(dataset))
    except Exception as exc:
        payload["len_error"] = type(exc).__name__
    try:
        sample = dataset[0]
    except Exception as exc:
        payload["sample0_error"] = type(exc).__name__
        payload["sample0_error_text"] = str(exc)[:500]
        return payload
    payload["sample0_type"] = type(sample).__qualname__
    if isinstance(sample, Mapping):
        keys = sorted(str(key) for key in sample.keys())
        payload["sample0_keys"] = keys
        payload["sample0_has_prefix_denoising_hybrid"] = (
            "prefix_denoising_hybrid" in sample
        )
        hybrid = sample.get("prefix_denoising_hybrid")
        payload["sample0_prefix_denoising_hybrid_type"] = (
            type(hybrid).__qualname__ if hybrid is not None else None
        )
    elif isinstance(sample, (list, tuple)):
        payload["sample0_len"] = len(sample)
        first = sample[0] if sample else None
        payload["sample0_first_type"] = type(first).__qualname__
        if isinstance(first, Mapping):
            keys = sorted(str(key) for key in first.keys())
            payload["sample0_first_keys"] = keys
            payload["sample0_first_has_prefix_denoising_hybrid"] = (
                "prefix_denoising_hybrid" in first
            )
            hybrid = first.get("prefix_denoising_hybrid")
            payload["sample0_first_prefix_denoising_hybrid_type"] = (
                type(hybrid).__qualname__ if hybrid is not None else None
            )
    return payload


__all__ = [
    "_resolve_text_vocab_size",
    "_resolve_embedding_rows",
    "_validate_batch_contract",
    "GradAccumLossScaleMixin",
]
