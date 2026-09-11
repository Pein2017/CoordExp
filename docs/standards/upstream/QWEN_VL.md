---
doc_id: docs.standards.upstream.qwen-vl
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Transformers Qwen-VL model, processor, and CoordExp integration notes.
updated: 2026-09-09
---

# Transformers Qwen-VL

Source scope: installed `transformers==4.57.1`, dense `qwen3_vl`.
Qwen2-VL and Qwen2.5-VL appear only as family-boundary reminders; their
execution is not specified by the pseudocode below. Resolve `${TRANSFORMERS_ROOT}` with:

```bash
python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)"
```

The local Qwen3-VL model source and installed version were checked on
2026-09-09. This is a version-scoped integration reference, not a claim about
the latest upstream release. Recheck the selected symbols when the runtime
changes. Local paths below refer to this checkout; bind research conclusions
to its commit and the actual loaded model.

## Architecture Map

- Qwen3-VL generated files say upstream edits belong in
  `modular_qwen3_vl.py`, not installed generated files. CoordExp must not patch
  local HF model files.
- `Qwen3VLConfig` composes vision and text configs and carries multimodal token
  ids: `vision_start_token_id`, `vision_end_token_id`, `image_token_id`, and
  `video_token_id`.
- `Qwen3VLForConditionalGeneration` wraps `Qwen3VLModel` and `lm_head`; the
  backbone exposes `model.visual` and `model.language_model`.
- Qwen3-VL vision uses patch embedding, learned/interpolated position
  embeddings, vision rotary embeddings, ViT blocks, a merger, and DeepStack
  merger outputs from configured vision layers.
- Qwen3-VL text is not pure text-only: DeepStack visual features can be injected
  into early hidden states through `visual_pos_masks` and
  `deepstack_visual_embeds`.
- Auto mappings keep Qwen2-VL, Qwen2.5-VL, and Qwen3-VL separate. Do not
  collapse their processor, RoPE, video, or attention behavior.

## Processor And Grid Rules

- `Qwen3VLProcessor` wraps an image processor, Qwen2 tokenizer, and Qwen3-VL
  video processor.
- Image placeholders expand from `image_grid_thw` using visual merge size.
  CoordExp should build messages through the processor/template path and then
  verify placeholder counts, not manually synthesize repeated image tokens.
- Qwen3 video placeholders differ from Qwen2.5: Qwen3 inserts text timestamps
  and frame blocks; Qwen2.5 uses `second_per_grid_ts` flows.
- Qwen3 `get_rope_index()` splits video grids into per-frame rows and uses text
  timestamps rather than absolute video time ids.
- Beam/generation expansion is visual-tensor aware because `pixel_values` is
  flattened across visual tokens and `image_grid_thw` is flattened across
  images, not ordinary batch-major metadata.

## Message-to-model tensor flow

All pseudocode in this page is conceptual, not a runnable API or a substitute
for the owning helpers. The detailed path is image-only. The local
`NativeRequest` accepts one image per request; upstream can represent multiple
images. Video requires timestamps/frame blocks and separate payloads. The
separate `qwen3_vl_moe` model family is outside this dense-model walkthrough.

Let `B` be text batch size, `L` the padded token length, `I` the total images,
`p` spatial patch size, `τ` temporal patch size, `m` spatial merge size,
`D_v` vision hidden width, `D` text hidden width, and `V` vocabulary size.
For image `i`, define `G_i = T_i H_i W_i` and `M_i = G_i / m²`.
Read values from the loaded processor and model configs. Vision config defaults
such as `p=16`, `τ=2`, `m=2`, and DeepStack indexes `[8,16,24]` do not establish
an arbitrary checkpoint's values.

| Boundary | Tensor shape / meaning |
| --- | --- |
| Tokenized chat | `input_ids`, `attention_mask`: `[B,L]`, including expanded image placeholders |
| Image grids | `image_grid_thw`: `[I,3]`; spatial dimensions are patch-grid counts before spatial merging |
| Pixel patches | `pixel_values`: `[sum(G_i), C*τ*p*p]`; flattened patch axis, not `[B,C,H,W]` |
| Vision backbone | `[sum(G_i), D_v]`; per-frame attention boundaries come from grids |
| Primary merger output | `[sum(M_i), D]`; `out_hidden_size` must match the text embedding width |
| Each DeepStack stream | `[sum(M_i), D]`; a list of streams, not a text batch tensor |
| Scatter / decoder states | `[B,L,D]`; boolean visual positions are `[B,L]` |
| Native MRoPE / packed positions | `[3,B,L]` / `[4,1,L_pack]` in the local packed owner |
| Selected logits | `[B,K,V]`; `K=L` for full logits, otherwise retained physical rows |

```text
messages + images in matching occurrence order
    -> checkpoint chat template + image processor + tokenizer
    -> [B,L] token IDs      + [sum(G_i), C*τ*p*p] patches + [I,3] grids
       -> text embeddings     -> ViT -> primary merger + DeepStack streams
       -> scatter primary visual embeddings into image-token positions
       -> decoder layers, with early post-layer DeepStack additions
       -> final norm -> select time rows -> lm_head -> [B,K,V]
```

### Chat rendering and image expansion

`Qwen3VLProcessor.__call__` receives rendered text; applying the checkpoint's
chat template is a separate step unless using a combined processor route.
[`src/qwen/encoding.py`](../../../src/qwen/encoding.py) renders a completed
training conversation with `add_generation_prompt=False`. A generation prompt
instead needs the appropriate assistant prefix from the actual template.
Appending arbitrary text to a completed assistant turn is not automatically
an exact continuation; use token-history replay for already observed actions.

```text
messages = role/content records containing image occurrences and text
chat = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=task_requires_prefix)
patches, grids = image_processor(images_in_occurrence_order,
                                do_resize=False, ...)
for each image placeholder in chat order across the batch:
    count = product(next_grid) / merge_size²
    expand that placeholder into count image-token markers
ids, mask = tokenizer(expanded_chat, padding=task_policy, ...)
assert number_of_image_tokens == sum(product(grids) / merge_size²)
```

The real processor uses temporary placeholders while expanding text, so newly
inserted image markers are not expanded again. The inspected `Qwen2VLImageProcessor` and `Qwen2VLImageProcessorFast`
implementations, available through `AutoImageProcessor`, arrange adjacent `m×m`
patches together for the merger. A still image is temporally repeated to fill
`τ`, giving `T=1`; the slow and fast preprocessors then flatten patch content.
The fast path groups equal image shapes for preprocessing and restores input
order before concatenation. Disabling resize preserves prepared geometry;
it does not disable rescaling/normalization or make arbitrary dimensions valid.

A text batch of two requests with one differently sized image each still has
`I=2`, but its pixel first dimension is `G_0+G_1`, not two. Reordering only
`image_grid_thw`, or applying an ordinary batch repeat to patches, breaks the
image/token correspondence. The model's `_expand_inputs_for_generation` has
visual-aware expansion; use the model's generation path rather than inventing
beam replication.

### Vision merger and DeepStack timing

Source owners: `Qwen3VLVisionPatchEmbed.forward`,
`Qwen3VLVisionModel.forward`, `Qwen3VLVisionPatchMerger.forward`,
`Qwen3VLModel.get_image_features`, and `Qwen3VLTextModel.forward`.

```text
z = patch_embed(patches)                         # [sum(G_i), D_v]
z += interpolated_learned_vision_positions(grids)
vision_rope, frame_boundaries = derive_from(grids)
deep = []
for j, vision_block in enumerate(vision_blocks):
    z = vision_block(z, vision_rope, frame_boundaries)
    if j is a configured DeepStack extraction index:
        deep.append(deep_merger_for(j)(z))        # [sum(M_i), D]
primary = final_merger(z)                         # [sum(M_i), D]
x = token_embedding(ids)                         # [B,L,D]
x = scatter_image_positions(x, primary)
for k, decoder_layer in enumerate(text_layers):
    x = decoder_layer(x, causal_attention, text_rope, cache)
    # Decoder-layer output hooks run here, before the following addition.
    if k < len(deep):
        x[visual_positions] = x[visual_positions].clone() + deep[k]
x = final_text_norm(x)
```

Extraction indexes name vision blocks; injection indexes are the first
`len(deep)` text layers. They are different coordinates. The primary merger
normalizes width `D_v` before grouping `m²` patches; DeepStack mergers use
post-group normalization at width `m²*D_v`. Both project to `out_hidden_size`.
`get_image_features` splits the primary output by image token counts while
leaving each DeepStack stream concatenated. Model forward concatenates the
primary features again to scatter them in token occurrence order.

No detach is inserted in this visual-to-text path. A differentiable forward
can propagate gradients through the primary scatter and DeepStack additions
to trainable vision components. Freezing parameters and disabling gradient
recording are different choices; see [autograd](#autograd-and-captured-states).

## Position IDs

- Qwen3-VL accepts 3-row MRoPE `position_ids` for temporal/height/width.
- If a 4-row tensor is supplied, row 0 is used as text positions for causal mask
  and text attention, while rows 1-3 are MRoPE geometry.
- During generation, Qwen3-VL recomputes position IDs with cached
  `rope_deltas` and drops visual tensors after the first cache step.
- Manual `position_ids` should be treated as a high-risk diagnostic override.
  If used, preserve either the 3-row MRoPE contract or the 4-row text-plus-MRoPE
  contract exactly.

### Prefill, cached decode, and MRoPE state

`get_rope_index` assigns text runs increasing positions in all three axes.
Image placeholders receive temporal/height/width grid coordinates, offset by
the preceding text; the next text run starts after the maximum visual
coordinate. Therefore the final position need not equal the token count.
Its `rope_deltas` is `[B,1]`: maximum assigned position plus one minus the
physical input-row length, including padding in that length calculation.

For the ordinary, noncompiled generation path with caching:

```text
# Prefill: uncached prompt tokens and visual payload are passed together.
position_ids, delta = model.get_rope_index(prompt_ids, grids, mask)
model.rope_deltas = delta
out = multimodal_forward(prompt, pixels, grids, position_ids, use_cache=True)
cache = out.past_key_values                       # per-layer key/value history

# Decode: GenerationMixin selects the uncached token(s) and updates cache_position.
while continuation_is_needed:
    kwargs = prepare_inputs_for_generation(history, cache, cache_position, ...)
    kwargs.position_ids = None                   # Qwen forward derives them
    if cache_position[0] != 0:
        kwargs.pixel_values = None
        kwargs.pixel_values_videos = None
    positions = expand_to_three_axes(
        arange(number_of_uncached_tokens) + cache_position[0] + model.rope_deltas)
    out = text_forward(uncached_tokens, positions, cache, ...)
    cache = out.past_key_values
```

This explains the branch, not an independent implementation of generation.
`Qwen3VLModel.forward` also handles absent/empty caches, explicit positions,
batch expansion, and a distinct compiled prefill heuristic. The generation
preparation override resets `position_ids` to `None`; passing manual positions
to `generate` is not equivalent to a direct forward with explicit positions.

During decode, no fresh DeepStack streams are created because pixels are not
forwarded. The prompt's multimodal effect remains in the decoder's cached
history; there is no repeated vision forward per generated token in this
cached route. `Qwen3VLTextAttention.forward` updates each layer's cache after
Q/K normalization and rotary application. `rope_deltas` is model-side state,
not a freestanding substitute for the matching token history, grid and KV
cache. A new independent history must derive its own positions. Local exact
replay removes stale cache/position fields and uses `use_cache=False`.

For local packed training, concatenate per-segment `[3,1,L_segment]` geometry
and prepend segment-local `arange` text rows. In `Qwen3VLTextModel.forward`,
the first row of a four-row input is routed to causal-mask/text-attention
handling, while the remaining three generate rotary embeddings. A valid
four-row tensor alone does not prove FA2 isolation; the local forward supplies
and validates its varlen plan separately.

## Logits

- Qwen3-VL forward accepts `logits_to_keep`.
- It slices hidden states before `lm_head`; `0` keeps all logits.
- CoordExp token diagnostics, coordinate scoring, and segment-aware sidecar
  remapping require full-vocabulary logits and either a full sequence time axis
  or an explicit physical-row map for compact selected-row logits.

## Local Execution Boundaries

Select the path by the actual execution task. A native multimodal sequence and
a concatenated training pack have different position and attention owners.

| Task | Owner and distinction |
| --- | --- |
| Native preparation or exact-history replay | [`src/qwen/native.py`](../../../src/qwen/native.py): `prepare_native_inputs` encodes image/text inputs through the processor. `exact_history_inputs` and `prepare_replay` retain the visual payload and derive positions through the loaded model's `get_rope_index`; `resolve_rope_index` locates that owner through wrapper `.model` levels. |
| Packed training forward | [`src/qwen/positions.py`](../../../src/qwen/positions.py): `build_qwen_position_inputs` constructs four rows, resetting text positions and deriving MRoPE independently per segment. [`src/qwen/forward.py`](../../../src/qwen/forward.py) owns visual concatenation, explicit FA2 arguments, `labels=None`, and `use_cache=False`. |
| Local supervision and objective | [`src/supervision/`](../../../src/supervision/) and [`src/losses/runner.py`](../../../src/losses/runner.py), governed by the [supervision/loss contract](../../../openspec/specs/coordexp-infras-supervision-losses/spec.md). Model-provided loss is not the local loss owner. |

### Packing and causal alignment

Calling the native whole-sequence position helper on a concatenated pack does
not establish segment isolation. Text reset points, MRoPE geometry, and FA2
segment boundaries must agree; use the
[packing/forward contract](../../../openspec/specs/coordexp-infras-packing-forward/spec.md)
and [`tests/qwen/test_positions.py`](../../../tests/qwen/test_positions.py)
for the existing reset and upstream per-segment parity checks.

For a prompt of length `P` followed by `N` target tokens, causal scores use
logit rows `P-1` through `P+N-2`. Selecting the last `N` rows of the full
prompt-plus-target forward is one token late. `prepare_replay` retains `N+1`
rows for compact replay; `ExactReplay.aligned_logits` selects the first `N`.
[`tests/qwen/test_native.py`](../../../tests/qwen/test_native.py) compares full
and compact scores and gradients and distinguishes the wrong shift. Packed
`logits_position_ids` instead map selected rows to physical positions; see
[`tests/qwen/test_forward.py`](../../../tests/qwen/test_forward.py). Neither
compaction nor packing changes the target-to-previous-logit relationship or
permits a target to use the previous segment's logits.

### Literal teacher forcing and selected logits

A replay forward supplies the complete observed prompt and target token IDs.
It does not sample new targets. Causal attention means the state at token
position `j` predicts token `j+1`, even though all target tokens are present
in the same forward.

```text
ids = concat(prompt_ids, observed_target_ids)      # length P+N
hidden = model_backbone(ids, visual_payload, positions, use_cache=False)
# Conceptual full-logit route:
full_logits = lm_head(hidden)                     # [1,P+N,V]
score_rows = full_logits[0, P-1:P+N-1, :]          # exactly N rows
# Equivalent compact route used by local prepare_replay:
kept_logits = lm_head(hidden[:, -(N+1):, :])       # [1,N+1,V]
score_rows = kept_logits[0, :N, :]
log_probs = log_softmax(score_rows.float(), dim=-1)
target_log_probs = gather(log_probs, observed_target_ids)
loss = explicit_local_reduction(-target_log_probs)
```

For example, `P=3, N=2` uses full rows `2,3` to score target positions `3,4`.
The final row `4` predicts the token after the supplied continuation and is
excluded. `logits_to_keep=1` is sufficient for the next generation decision,
but not for scoring both observed targets. Integer `logits_to_keep=0` uses
`slice(0,None)` in upstream forward, retaining all rows; a tensor argument
selects explicit row indices before `lm_head`. This saves vocabulary projection
work/storage; it does not skip earlier decoder computation.

Upstream `Qwen3VLForConditionalGeneration.forward` optionally invokes
`self.loss_function` when `labels` are supplied. The causal loss implementation
shifts labels, including ignore handling; it is not aware of local token atoms,
packed segment ownership, or the planned-step segment-balanced reducer.
Local packed forward consequently passes `labels=None` and returns logits to
its explicit loss owner. Native replay also omits labels and aligns observed
action scores explicitly. A plain token mean in this pseudocode must not be
substituted for the configured training reduction.

### Autograd and captured states

| Operation | Execution consequence |
| --- | --- |
| `model.eval()` | Sets evaluation behavior such as dropout; does not disable autograd. |
| `torch.no_grad()` | Suppresses recording inside the context; a later operation outside it can use ordinary resulting tensors in a new graph. It does not recover the suppressed graph. |
| `torch.inference_mode()` | Additionally creates inference tensors with restrictions relevant to later autograd; use only when no derivative through that execution is required. |
| `tensor.clone()` | New storage while preserving a recorded gradient connection to the original. |
| `tensor.detach().clone()` | Independent value snapshot with no gradient connection to the original computation. |

The installed `GenerationMixin.generate` is decorated with `torch.no_grad()`.
For gradients of observed-token scores, use a differentiable replay forward;
requesting scores from that generation call does not restore its graph.

An integer position tensor can still be saved for backward by an operation
involving trainable tensors. `requires_grad=False` alone does not make an
inference tensor safe for a later differentiable forward. In
[`src/qwen/native.py`](../../../src/qwen/native.py), `derive_position_ids`
returns an ordinary detached clone under `torch.inference_mode(False)`;
`move_to_device` also converts inference tensors this way. The native tests
above exercise differentiable positions and replay. Choose gradient mode for
the intended measurement; a detached capture cannot support gradients through
its original computation.

In the inspected 4.57.1 `Qwen3VLTextModel.forward`, a decoder layer returns
before `_deepstack_process` adds visual features at selected image positions.
That helper writes into `hidden_states` in place. Therefore a decoder-layer
output hook observes the pre-addition boundary, and retaining its tensor by
reference can later expose the post-addition values. Clone inside the hook to
preserve that boundary (`detach().clone()` for a value-only snapshot; `clone()`
when retaining the gradient path). Name the capture boundary explicitly.
The maintained capture owners are
[`src/qwen/inspection.py`](../../../src/qwen/inspection.py), with mutation and
exception-cleanup counterexamples in
[`tests/qwen/test_inspection.py`](../../../tests/qwen/test_inspection.py).
`CaptureHiddenRows` deliberately detaches its diagnostics; `CaptureInputs`
also detaches its snapshots, which can be inputs to a new functional replay.
Neither preserves gradients through the original captured computation. A
direction needing that derivative must retain a graph-connected hook value
at its declared boundary, rather than treating these snapshot helpers as
differentiable capture APIs.
The mutation order is owned by `Qwen3VLTextModel.forward` and
`Qwen3VLTextModel._deepstack_process` in the installed model source below.

## CoordExp Rules

- Keep `processor_do_resize=false`. HF Qwen image processors can default to
  resizing; CoordExp geometry assumes offline-prepared images and aligned
  coordinates.
- Preserve `pixel_values`, `image_grid_thw`, `pixel_values_videos`,
  `video_grid_thw`, `attention_mask`, and cache state together. Do not reorder
  grids independently of token order.
- For LoRA/freezing, inspect loaded checkpoint names. Common Qwen3-VL prefixes:
  `model.visual.blocks.*`, `model.visual.merger.*`,
  `model.visual.deepstack_merger_list.*`, and
  `model.language_model.layers.*`.
- Make `attn_implementation` explicit in run metadata, especially when comparing
  `eager`, `sdpa`, and `flash_attention_2`.

## Troubleshooting

- `qwen3_vl` import or AutoModel errors usually indicate an older Transformers
  install.
- Image token/features mismatch: inspect `image_grid_thw`, merge size, chat
  template image-token count, and placeholder-mask construction.
- Duplicate visual work in generation: inspect `cache_position`; Qwen3-VL should
  only forward pixels on the first cache step.
- Video temporal drift: record fps metadata, Qwen3 timestamp insertion, frame
  sampling, and `do_resize`.
- Unexpected `token_type_ids`: Qwen3-VL examples may drop them; local processor
  defaults do not require returning them.

## Handles

Inspected symbols in installed Transformers 4.57.1:

| Source relative to `${TRANSFORMERS_ROOT}` | Symbols / scope |
| --- | --- |
| `models/qwen3_vl/configuration_qwen3_vl.py` | `Qwen3VLVisionConfig`, `Qwen3VLTextConfig`, `Qwen3VLConfig`; defaults versus nested checkpoint config |
| `models/qwen3_vl/processing_qwen3_vl.py` | `Qwen3VLProcessor.__call__`; image occurrence expansion, video timestamp branch, tokenizer dispatch |
| `models/qwen2_vl/image_processing_qwen2_vl.py` | `Qwen2VLImageProcessor._preprocess`, `preprocess`; slow patch/grid layout |
| `models/qwen2_vl/image_processing_qwen2_vl_fast.py` | `Qwen2VLImageProcessorFast._preprocess`; grouped fast patch/grid layout |
| `models/qwen3_vl/modeling_qwen3_vl.py` | `Qwen3VLVisionPatchEmbed.forward`, `Qwen3VLVisionPatchMerger.forward`, `Qwen3VLVisionModel.forward`; visual path |
| same model file | `Qwen3VLModel.get_image_features`, `get_rope_index`, `forward`; scatter and position/cache state |
| same model file | `Qwen3VLTextModel.forward`, `_deepstack_process`, `Qwen3VLTextAttention.forward`; injection, mask, rotary and KV update |
| same model file | `Qwen3VLForConditionalGeneration.forward`, `prepare_inputs_for_generation`, `_expand_inputs_for_generation`; selected logits and generation boundaries |
| `generation/utils.py`, `loss/loss_utils.py` | `GenerationMixin.generate` gradient decorator; `ForCausalLMLoss` causal shift |

For a targeted reread without loading weights or using a GPU:

```bash
python - <<'SOURCE'
import inspect
import transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
print(transformers.__version__)
owner = qwen.Qwen3VLModel.get_rope_index  # select the symbol needed above
print(inspect.getsourcefile(owner))
print(inspect.getsource(owner))
SOURCE
```

For a loaded runtime, also inspect `type(processor.image_processor)` and
`model.config.vision_config` / `model.config.text_config`; the processor uses
Auto classes, so a package's available class or config default does not prove
what a particular checkpoint loaded. The pixel shape table follows executable
preprocessing and patch embedding: the generic four-dimensional image shape
in the installed `get_image_features` docstring does not describe this flattened
input boundary. This walkthrough does not validate video execution, MoE,
compiled generation, or another attention backend for a research run.

- Local config:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/configuration_qwen3_vl.py`
- Local model:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/modeling_qwen3_vl.py`
- Local processor:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/processing_qwen3_vl.py`
- Local video processor:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/video_processing_qwen3_vl.py`
- Upstream docs:
  [Qwen3-VL v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl)
- Upstream source:
  [qwen3_vl tag v4.57.1](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/qwen3_vl)
