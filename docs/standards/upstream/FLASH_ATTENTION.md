---
doc_id: docs.standards.upstream.flash-attention
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: FlashAttention v2 and Transformers attention interface notes.
updated: 2026-06-07
---

# FlashAttention v2 / Transformers Attention Interface

Source scope: local `flash_attn==2.8.3`, `transformers==4.57.1`,
`torch==2.9.1+cu128`, `triton==3.5.1`. Runtime check on 2026-06-07 reported
CUDA available on `NVIDIA A100 80GB PCIe`, compute capability `(8, 0)`.

## Architecture Map

- `flash_attn` exports public APIs from `flash_attn_interface.py`: dense
  `flash_attn_func`, packed QKV/KV functions, varlen functions, and KV-cache
  functions.
- Unless `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, the installed package imports
  the compiled CUDA extension `flash_attn_2_cuda`.
- FA2 kernels are wrapped as `torch.library.custom_op` when supported by torch,
  which matters for `torch.compile`.
- Dense FA2 expects Q/K/V shaped `(batch, seqlen, heads, head_dim)`.
- Varlen FA2 expects flattened token-major Q/K/V shaped
  `(total_tokens, heads, head_dim)` plus:

```text
cu_seqlens_q: int32 tensor, shape [num_sequences + 1]
cu_seqlens_k: int32 tensor, shape [num_sequences + 1]
max_seqlen_q: Python int
max_seqlen_k: Python int
```

- `bert_padding.py` owns unpad/repad helpers for padded and concatenated
  sequences.
- Transformers routes attention through `AttentionInterface`; the registry maps
  `flash_attention_2` and `flash_attention_3` to the FlashAttention wrapper.
- Qwen3-VL declares FlashAttention and SDPA support. Its vision attention path
  passes `cu_seq_lens_q`, `cu_seq_lens_k`, and max lengths when FA2 is active.

## CoordExp Rules

- Prefer `attn_implementation: flash_attention_2` or
  `model.set_attn_implementation("flash_attention_2")` through Transformers or
  ms-swift configuration. Do not monkeypatch attention modules.
- Use `float16` or `bfloat16`; Transformers warns on unset or unsupported
  dtypes.
- Treat packed/padding-free training as a varlen attention contract.
- For CoordExp packing, the collator/batch builder should own explicit
  `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`.
  Transformers can infer from `position_ids`, but its own utility notes that
  collator-stage cumulative lengths are preferable.
- Do not rely on an ordinary 2D `attention_mask` alone to represent multiple
  packed examples inside one row. In Transformers FA2, a non-null
  `attention_mask` enters the padded unpad/repad path before padding-free packed
  logic.
- Preserve segment boundaries exactly. Wrong cumulative lengths permit
  cross-example attention or hide valid tokens, silently changing the training
  problem.
- If using `position_ids` inference, note the local packed-sequence detector is
  a heuristic for flattened `batch_size == 1` packed sequences. Multi-row
  packing should pass explicit varlen kwargs.
- For debugging attention scores, switch to `eager`; FA2 does not support
  `output_attentions=True` in the normal Transformers wrapper path.

## Determinism And Compile

- FA2 forward is deterministic; deterministic backward is opt-in, slower, and
  more memory hungry.
- Transformers can pass `deterministic` or use
  `FLASH_ATTENTION_DETERMINISTIC=1`.
- `max_seqlen_q/k` must be Python ints. Transformers calls `.item()` and notes
  that this can graph-break under `torch.compile` unless scalar-output capture
  is enabled.

## Common Failures

- Import failure: check Python, Torch, CUDA, GPU arch, wheel compatibility, and
  the compiled `flash_attn_2_cuda` extension.
- CPU-only run or model left on CPU: FA2 will warn or fail.
- Unsupported model class: `attn_implementation="flash_attention_2"` fails if
  the model class does not declare support.
- BetterTransformer: incompatible with FA2 in local Transformers checks.
- Zero-length tensors: local wrapper raises when query has any zero dimension.
- Causal-mask alignment: current FA2 uses bottom-right causal alignment for
  unequal query/key lengths; do not assume legacy top-left behavior.
- `return_attn_probs`: testing-only in FA2 and not a stable debug surface.

## Handles

- Local API exports:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/flash_attn/__init__.py`
- Local FA2 interface:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py`
- Local padding helpers:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/flash_attn/bert_padding.py`
- Transformers utility:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py`
- Transformers wrapper:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/integrations/flash_attention.py`
- Transformers support checks:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/modeling_utils.py`
- Upstream FA2 tag:
  [Dao-AILab/flash-attention v2.8.3](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3)
- Upstream attention docs:
  [Transformers attention interface v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)
