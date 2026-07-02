# CoordExp-Swift Training Infrastructure Rebuild Design

## Context

CoordExp-swift is a clean rebuild of the CoordExp training infrastructure in
this worktree. The old `openspec/` tree is archived under
`reference/legacy_openspec_2026-06-29/` as reference-only material. The current
design baseline is `DECISIONS.md`, `BLUEPRINT.md`, and this OpenSpec change.

The V1 target is Qwen3-VL single-image supervised training. Transformers,
PyTorch, PEFT, Accelerate, DeepSpeed, and flash-attn remain valuable external
substrates. MS-Swift remains a source-study reference, not a runtime dependency
or current contract authority.

The rebuilt repository owns the CoordExp-specific flow:

Raw Dataset -> Dataset Processing -> Chat Template -> Packing -> Tokenization
-> Visual Processing -> Model Construction -> Forward Pass -> Loss Computation
-> Backward Pass -> Distributed Training -> Checkpoint Management -> Eval.

The user's tradeoff order is accuracy and precision first, training and system
efficiency second, simplicity third, and extension capability fourth.

## Goals / Non-Goals

Goals:

- define the stable V1 contracts for config, data, template, Qwen encoding,
  packing, supervision, losses, adapters, embeddings, optimizer grouping,
  training runtime, artifacts, metrics, checkpoints, and forward eval;
- keep the entire teacher-forced training flow readable inside this repository;
- require source-study gates where upstream behavior is delicate, especially
  DoRA, special-token embeddings, Qwen3-VL processor behavior, MRoPE, and
  FlashAttention varlen boundaries;
- preserve a strict packed single-sequence training path with no padding;
- make the first vertical smoke a real five-planned-step Qwen3-VL train/eval
  acceptance path using the intended DoRA adapter surface after its source
  study passes.

Non-goals:

- no source implementation during this drafting and review loop;
- no attempt to preserve the archived OpenSpec tree as active authority;
- no old `src/` compatibility shims;
- no runtime dependence on MS-Swift;
- no rollout training, hidden-state losses, hidden-state/KV/runtime feature
  caches, video, multi-image, vLLM, exact optimizer/RNG resume, or DeepSpeed
  production claim in the first vertical smoke;
- no base-only or standard-LoRA pre-smoke unless the user makes a new decision.

## Decisions

### Authority And Ownership

The active OpenSpec home starts fresh. Archived legacy specs and archived
changes are historical reference only. This change defines new capabilities
instead of modifying the old capability names.

The active source root will be `src/` directly, with no extra `coordexp`
package layer. Module ownership follows the proposal docs: `src/data`,
`src/templates`, `src/qwen`, `src/packing`, `src/supervision`, `src/losses`,
`src/optim`, `src/training`, `src/runtime`, `src/artifacts`, `src/metrics`, and
`src/eval`.

### Config And Runtime

Training is config-first. The strict resolved config is the public contract.
Config inheritance is allowed, but the run artifact preserves the final
resolved YAML and JSON as self-contained evidence. Inheritance is intentionally
simple: one top-level parent per file, parent-first dictionary merge, list
replacement, no null deletion, file-local path origins, and cycle failures with
the full chain. Public configs specify `training.effective_batch_size` and
either production `epochs` or debug `max_steps`; runtime derives the
accumulation count from world size and writes that value only into runtime
receipts.

Planned steps are the schedule source of truth. Eval, checkpoint, logging, and
final events are resolved before training and are not retimed by recoverable
warnings or skipped unsafe optimizer updates.
Epoch-led runs use deterministic tail-fill for incomplete final
effective-batch windows. V1 never silently drops final packs and never performs
a smaller partial final optimizer update.

Packed training pins Qwen runtime controls up front: FlashAttention 2 is the
normal attention implementation, bf16/fp16 is required for that path, and setup
estimates worst-case full-sequence logits memory from
`packing.global_max_length`, vocab size, and dtype before model mutation. The
default forward may materialize only selected supervised rows when it records
the physical position map consumed by `LossContext`; this does not relax the
preflight budget guard.

`TrainRuntime` is Accelerate-first and owns backend mechanics. DeepSpeed
configuration may be schema-accepted and conflict-validated in V1, but
production support is not claimed until a separate systems smoke proves
prepare, backward, clipping, optimizer stepping, checkpoint save/load, and
rank-safe artifacts.

### Data, Template, And Qwen Encoding

The canonical example chain is `RawExample -> RenderedExample -> EncodedExample
-> PackedSequence`. Use `example`, not `sample`, for the stable semantic unit.

`src/data` validates JSONL examples, image paths, image dimensions, and bbox
coordinate-bin values. `src/templates` renders English-only Qwen chat messages,
prompt text, `supervised_response_text`, object ordering, and typed character
spans. `src/qwen` owns all interaction with `AutoProcessor`, tokenizer,
processor image handling, Qwen3-VL model loading, visual payloads, MRoPE helper
behavior, and model-output contracts.

The assistant target supervises answer content plus `<|im_end|>`. The following
newline in the Qwen `<|im_end|>\n` convention is ignored text. The first
supervised token is exactly the first assistant-answer token after the assistant
start boundary. Prompt, user, system, role/header, image placeholder, and other
control tokens are not supervised. Qwen setup must preflight that
`<|im_end|>\n` tokenizes as the `<|im_end|>` transition token followed by a
separate newline token.

Training uses no-resize image processing. The actual call path must use
`do_resize=False`, and packed cost uses actual no-resize `image_grid_thw` or a
proven-equivalent local computation. No-resize validation must derive
admissible dimensions from the loaded processor's `patch_size` and `merge_size`,
record those values in receipts, and enforce explicit raw-pixel and
merged-visual-token budget caps before processor reshape or model forward.

### Packing And Forward

The standard supervised forward path is one physical packed sequence per
rank/step, usually tensor shape `[1, L]` for Transformers compatibility. There
is no padded conceptual batch in V1 supervised training. Multiple encoded
examples may be concatenated under `packing.global_max_length`, but each
segment remains isolated for attention, position ids, supervision, metrics, and
loss accounting.

Packing cache reuse is first-class infrastructure, not an experimental future
cache. For a fixed dataset, template, Qwen encoding identity, processor
identity, object-ordering policy, and `packing.global_max_length`, the
implementation should pack once and then reuse the cache on later runs. Cache
miss materialization must use a forced default of 16 CPU workers so full
production JSONL packing is not serialized through one Python process. The
worker count is operational metadata: it must be recorded in cache receipts or
manifests, but it must not change the semantic cache fingerprint.

Packing owns physical placement and invertible mapping from logical token spans
to packed token positions. Qwen forward receives repo-built Qwen forward inputs
and uses Transformers for model internals, visual tower, visual replacement, and
LLM tower execution. CoordExp does not pass `inputs_embeds` in V1 and does not
use the model-side CE path.

Packed Qwen position inputs are deliberately segment-local. For packed
training, CoordExp builds the 4-row HF Qwen boundary shape `[text,t,h,w]` from
per-segment MRoPE computation, then concatenates the rows. Text-position reset
points must match the `PackedSegment` table and FlashAttention cumulative
sequence lengths; running an upstream helper once over the whole packed row is
not sufficient because it can produce continuous positions across segments.

FlashAttention varlen behavior must be proven with explicit varlen inputs such
as cumulative sequence lengths and max lengths. A 2D zero mask over a packed row
is not sufficient evidence of segment isolation.

### Supervision And Losses

`TokenAtom -> TokenSpan -> TokenSequence` is the canonical supervision
hierarchy. Dense labels are derived compatibility artifacts only. The loss
context owns causal shifting from `target_position` to `logits_position`.

The protected default losses are `BaseTokenCE` and `TokenTypeGateLoss`.
Both use selected logits upcast to fp32 for objective math. Base CE remains
full-vocabulary CE. Gate loss uses resolved vocabulary groups to penalize
token-type illegality for `desc_text`, `schema`, `coordinate`, and `eos`. The
gate formula is group-mass CE:
`logsumexp(all_logits) - logsumexp(allowed_group_logits)`. Every V1
`TokenAtom` must resolve to exactly one token type before protected losses run.
These V1 protected losses are the new baseline objective, not a parity claim
for archived coordinate soft-CE or object/role-balanced objective semantics.

Loss normalization is planned-step-window based and length invariant. The
protected V1 token-wise reducer is `segment_balanced`: mean over eligible atoms
within each segment, then mean over eligible segments across the complete
planned optimizer-step window. The loss runner must avoid backend double
scaling and must emit weighted per-term loss metrics plus top-level `acc_top1`
and `acc_top5`.

Finite handling is split into a pre-backward scalar finite gate and a
post-backward gradient/overflow gate. Recoverable bad examples or warnings may
be recorded, but an unsafe non-finite loss or gradient must not become a
corrupted optimizer update.

### Adapters, Embeddings, And Optimizer

The first adapter-enabled smoke uses DoRA, but `adapter.type: dora` must not
validate until the PEFT DoRA/`use_dora` source study and a minimal round-trip
probe pass. The source study selected `dora` as the public name; `dlora` is not
a V1 schema value.

Special-token embeddings are fully trainable selected-token deltas, not LoRA on
the embedding/head. The trainable group includes the four schema wrappers and
`<|coord_0|>` through `<|coord_999|>`. The implementation must compare custom
Qwen wrappers, PEFT `TrainableTokens`, and LoRA `trainable_token_indices`
before coding.

Optimizer groups are explicit. Every trainable parameter must match exactly one
approved LR/WD group or fail fast. Qwen tower groups include vision, aligner,
language, adapter parameters, and selected special-token embedding deltas.
In V1, vision/aligner/language are semantic adapter-target namespaces, not
permission to fine-tune full base-model parameters.

### Trainer, Artifacts, And Smoke

`SupervisedTrainer` owns the execution loop and calls approved components; it
does not own objective math, adapter taxonomy, Qwen internals, or artifact
schemas. `TrainRuntime` owns device/distributed/backend mechanics, rank guards,
backward, clipping, optimizer/scheduler stepping helpers, and distributed
gathering.

The artifact manager owns run directories, resolved configs, run manifests,
`resolved_step_schedule.json`, debug receipts, metrics, checkpoint metadata,
aliases, and eval.forward summaries. The first implementation must preserve
minimum key sets for the run manifest, metric events, eval.forward summaries,
and checkpoint metadata. Checkpoint names use unpadded planned-step ids and
`checkpoints/checkpoint-final.json` must always exist after a completed run.
`eval.forward` requires an explicit eval data source or explicit smoke-fixture
eval binding; train JSONL is not reused implicitly.

The first vertical smoke is the acceptance target: real
`packing.global_max_length`, sample-limited permanent fixture, five planned
steps, two forward-eval steps, protected losses, optimizer boundary, metrics,
checkpoint metadata, `resolved_step_schedule.json`, and final checkpoint alias.
The fixture contains at least two single-image examples and the single-rank
smoke uses `training.effective_batch_size: 2` by default so accumulation,
multi-segment packing, MRoPE resets, FA2 splits, and `segment_balanced`
denominators are exercised.

## Risks / Trade-offs

- Reimplementing the training pipeline increases near-term work, but reduces
  future context fragmentation and makes research losses easier to audit.
- Depending on Transformers internals for Qwen3-VL preserves model correctness,
  but requires careful source-study gates around processor, MRoPE, visual
  replacement, and FlashAttention behavior.
- DoRA-first smoke protects the intended production path, but it delays the
  first adapter-enabled smoke until the source study and probe pass.
- Strict no-padding packing is efficient and aligned with the user's training
  style, but it makes attention isolation and position-id contracts more
  important.
- Compact special-token embedding deltas are simpler to load with base-plus-
  adapter composition than full embedding exports, but the tied-head behavior
  must be proven for Qwen3-VL.
- The rebuild archives old `src/` as reference-only, but implementation must
  inventory legacy correctness invariants before discarding its tests as an
  active safety net.

## Migration Plan

1. Complete this OpenSpec artifact set: proposal, design, seven delta specs,
   and tasks.
2. Run OpenSpec validation and hygiene checks.
3. Run four read-only review lanes: contract/spec, Qwen/upstream,
   training/loss/runtime, and artifact/config/smoke.
4. Patch accepted P0/P1 findings; patch P2 only when cheap and aligned with
   accuracy, efficiency, and simplicity.
5. Stop at `ready for user approval`.
6. After user approval, implementation begins with the source-study gates,
   smoke fixture pinning, legacy `src/` archive, fresh skeleton, and then the
   approved vertical slice.

## Open Questions

There are no user-blocking architecture questions for this drafting pass.
Implementation remains blocked on explicit approval and the following
source-study gates:

- DoRA definition/source study and round-trip probe;
- special-token embedding mechanism study;
- Qwen3-VL no-resize, MRoPE, and FlashAttention varlen source verification;
- DeepSpeed systems smoke before production support is claimed.
