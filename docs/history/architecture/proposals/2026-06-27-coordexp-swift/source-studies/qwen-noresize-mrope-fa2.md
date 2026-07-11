# Qwen No-Resize, MRoPE, And FlashAttention Source Study

## Scope

Wave 1A read-only study for OpenSpec tasks 2.6, 2.7, and 2.8. Wave 1B later
added executable Qwen processor/forward and FA2 varlen probe receipts recorded
below. No production `src/` code was implemented by this study/probe gate.

## Verified Local Facts

Local model:

```text
/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

Package facts:

```text
torch 2.9.1+cu128
transformers 4.57.1
peft 0.17.1
flash-attn 2.8.3
```

Model facts:

```text
architectures: Qwen3VLForConditionalGeneration
model_type: qwen3_vl
dtype: torch.float32
tie_word_embeddings: true
text_config.vocab_size: 152670
text_config.hidden_size: 2048
```

Processor facts:

```text
processor: Qwen3VLProcessor
image_processor: Qwen2VLImageProcessorFast
patch_size: 16
merge_size: 2
temporal_patch_size: 2
```

Token identity:

```text
<|im_end|>       -> [151645]
<|im_end|>\n     -> [151645, 198]
<|coord_0|>      -> 151670
<|coord_999|>    -> 152669
```

## No-Resize Contract

For the local model, no-resize spatial admissibility requires:

```text
height % (patch_size * merge_size) == 0
width  % (patch_size * merge_size) == 0
```

With `patch_size=16` and `merge_size=2`, the required factor is 32.

Do not hardcode factor 14 or 28. Derive the factor from the loaded processor.

Observed lightweight CPU processor behavior:

```text
64 x 96, do_resize=False
  image_grid_thw = [[1, 4, 6]]
  pixel_values.shape = (24, 1536)
  merged visual tokens = 1 * 4 * 6 // 2**2 = 6
  prompt expansion inserts 6 <|image_pad|> tokens

65 x 96, do_resize=False
  fails with an opaque reshape RuntimeError
```

Therefore, V1 must pre-check before processor call:

- image id/path;
- decoded height and width;
- loaded `patch_size`;
- loaded `merge_size`;
- required factor;
- raw pixels;
- raw-pixel budget;
- expected `image_grid_thw`;
- expected raw patch rows;
- expected merged visual-token count;
- merged visual-token budget.

Do not use Qwen2VLImageProcessorFast `get_number_of_image_patches()` blindly for
no-resize planning, because that helper uses smart-resize logic and is not
equivalent to no-resize pack-cost planning.

## MRoPE Contract

V1 packed Qwen training should pass explicit 4-row `position_ids` at the Qwen
boundary:

```text
[text, t, h, w]
shape: [4, batch, seq]
```

Installed Qwen3-VL source supports that boundary:

- default 3-row positions are temporal/height/width;
- when a 4-row tensor is supplied, row 0 is split as text position ids and rows
  1-3 feed rotary MRoPE.

Do not call upstream `get_rope_index` once over a whole packed row. Installed
`get_rope_index` advances positions continuously using the previous max
position. That is correct for one semantic example, but it silently violates
CoordExp packed segment isolation if applied to a multi-segment physical row.

Recommended V1 construction:

1. For each `PackedSegment`, compute or mimic Qwen MRoPE positions from that
   segment's expanded `input_ids` and `image_grid_thw`.
2. Add a segment-local text-position row.
3. Concatenate all four rows after per-segment construction.
4. Assert row count and row meaning.
5. Assert text position reset at every `PackedSegment.start_position`.
6. Assert reset points equal FA2 cumulative-sequence splits.

`rope_deltas` are generation/cache metadata for implicit-position paths. V1
training uses `use_cache=False` and explicit position ids, so `rope_deltas`
should not be treated as the source of truth for packed training.

## FlashAttention Varlen Contract

The local Qwen3-VL model supports FlashAttention backends, and
`is_flash_attn_2_available()` is true in the environment. However, the local
config resolves `_attn_implementation=None` and model dtype is fp32. Therefore,
packed training smoke must explicitly resolve:

```text
attn_implementation = flash_attention_2
dtype in {bf16, fp16}
```

Transformers FA2 dispatch expects fp16 or bf16. A valid smoke must not rely on
the local fp32 config default.

The branch proof must show the padding-free varlen path, not the padding-mask
unpad path. A 2D zero mask over a packed row is not segment-isolation evidence.

Expected V1 proof:

- pass explicit `cu_seq_lens_q`;
- pass explicit `cu_seq_lens_k`;
- pass `max_length_q`;
- pass `max_length_k`;
- derive all of the above from `PackedSegment` boundaries;
- hook or wrap the installed Transformers/flash-attn call in a tiny probe;
- record the actual branch and kwargs in a receipt.

Receipt should include:

- segment count;
- segment boundaries;
- cumulative sequence lengths;
- max lengths;
- resolved attention implementation;
- model dtype;
- autocast state;
- whether branch evidence came from explicit varlen kwargs.

## Required Wave 1B Probe Shape

Create `scripts/probes/coordexp_swift/qwen_processor_forward_probe.py` and
`scripts/probes/coordexp_swift/fa2_varlen_probe.py` later in Wave 1B.

The probes should be tiny and use the smallest valid resource footprint:

- local-only processor/model loading;
- valid 32-divisible synthetic or smoke image;
- `do_resize=False`;
- `attn_implementation="flash_attention_2"`;
- `torch.bfloat16` or `torch.float16`;
- `labels=None`;
- `use_cache=False`;
- no `inputs_embeds`;
- explicit 4-row per-segment `position_ids`;
- explicit FA2 varlen kwargs or installed-equivalent padding-free varlen branch
  evidence;
- receipt assertion that `attention_mask is None`, the observed branch is
  padding-free varlen, all `cu_seq_lens_q/k` and `max_length_q/k` values are
  non-null, and those values match `PackedSegment` cumulative boundaries;
- JSON receipt such as `qwen_forward_contract.json`.

## Findings

- **P0:** Whole-pack `get_rope_index` would silently violate packed segment
  isolation. V1 must compute MRoPE per segment and concatenate after reset.
- **P0:** FA2 segment isolation is unproven unless the actual padding-free
  varlen branch is observed. A 2D mask is insufficient.
- **P1:** No-resize invalid dimensions currently fail as an opaque processor
  reshape error. V1 needs processor-derived pre-checks.
- **P1:** Local model config is fp32 while FA2 expects fp16/bf16. Smoke must
  resolve dtype deliberately.
- **P2:** No-resize pack-cost planning should use actual processor
  `image_grid_thw` or a proven equivalent formula, not smart-resize helpers.

## Wave 1B Probe Evidence

Executed from `/data/CoordExp/.worktrees/CoordExp-swift` on 2026-06-30:

```bash
python scripts/probes/coordexp_swift/qwen_processor_forward_probe.py \
  --output-dir outputs/probes/coordexp_swift/qwen_processor_only
CUDA_VISIBLE_DEVICES=6 python scripts/probes/coordexp_swift/qwen_processor_forward_probe.py \
  --run-model-forward --device cuda --dtype bfloat16
python scripts/probes/coordexp_swift/fa2_varlen_probe.py \
  --device cpu --dtype bfloat16 \
  --output-dir outputs/probes/coordexp_swift/fa2_varlen_cpu
CUDA_VISIBLE_DEVICES=6 python scripts/probes/coordexp_swift/fa2_varlen_probe.py \
  --device cuda --dtype bfloat16 \
  --output-dir outputs/probes/coordexp_swift/fa2_varlen_cuda
```

Receipts:

- `outputs/probes/coordexp_swift/qwen_processor_only/qwen_forward_contract.json`;
- `outputs/probes/coordexp_swift/qwen_processor_forward/qwen_forward_contract.json`;
- `outputs/probes/coordexp_swift/fa2_varlen_cpu/fa2_varlen_probe_receipt.json`;
- `outputs/probes/coordexp_swift/fa2_varlen_cuda/fa2_varlen_probe_receipt.json`.

Observed Qwen processor/forward contract:

- processor class: `Qwen3VLProcessor`;
- image processor class: `Qwen2VLImageProcessorFast`;
- no-resize synthetic image: 64 x 96;
- `patch_size: 16`, `merge_size: 2`, `temporal_patch_size: 2`;
- required no-resize spatial factor: 32;
- `image_grid_thw`: `[[1, 4, 6]]`;
- raw patch rows: 24;
- pixel shape: `[24, 1536]`;
- merged visual tokens and expanded `<|image_pad|>` count: 6;
- `<|im_end|>\n` tokenizes as `<|im_end|>` plus newline token id 198;
- opt-in model forward used `labels=None`, `use_cache=False`, no
  `inputs_embeds`, bf16, and two packed image/text segments;
- packed segment lengths: `[21, 22]`;
- packed segment boundaries: `[0, 21, 43]`;
- 4-row position ids shape: `[4, 1, 43]`;
- text-position reset values at segment starts: `[0, 0]`;
- per-segment position summaries record all four rows:
  `[text, temporal, height, width]`;
- whole-pack `get_rope_index` contrast shows the disallowed whole-pack helper
  would start the second segment at text position `18`, not `0`;
- loaded Qwen attention implementation was `flash_attention_2`;
- integrated Qwen forward with monkeypatched installed FA2 import observed a
  real Qwen-routed `padding_free_varlen` text-attention call with
  `cu_seq_lens_q == cu_seq_lens_k == [0, 21, 43]` and
  `max_length_q == max_length_k == 22`;
- no ordinary flash path, pad path, or unpad path was reached for the matching
  text-attention calls;
- logits shape matched `[1, 43, 152670]`.

Observed FA2 contract:

- installed `transformers` version: 4.57.1;
- installed `flash_attn` version: 2.8.3;
- explicit varlen branch source fragments were present in
  `transformers/modeling_flash_attention_utils.py`;
- observed branch: `padding_free_varlen`;
- `attention_mask: null`;
- standalone synthetic cumulative sequence lengths: `[0, 2, 5, 6]`;
- `max_length_q` and `max_length_k`: 3;
- all branch assertions passed on CPU and on CUDA bf16.

## Status

Tasks 2.6, 2.7, and 2.8 have source-study evidence and Wave 1B probe receipts
for no-resize behavior, packed Qwen MRoPE boundary construction, and FA2
explicit-varlen branch selection. Production implementation remains blocked by
task 2.10 review and the later OpenSpec implementation approval gate.
