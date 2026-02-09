# stage2-ab-training Specification

## Purpose
TBD - created by archiving change 2026-01-27-add-stage2-ab-iterative-softctx-bboxonly. Update Purpose after archive.
## Requirements
### Requirement: Stage-2 AB trainer variant is selectable via YAML
When training config sets `custom.trainer_variant: stage2_ab_training`, the system SHALL use the Stage-2 AB trainer implementation.

The trainer MUST be configurable via YAML and MUST NOT require new CLI flags.

Canonical config location (typed):
- Stage-2 AB knobs MUST be expressed under the top-level `stage2_ab` mapping (parallel to `training` and `custom`).
- Unknown keys under `stage2_ab` MUST fail fast with actionable guidance (to avoid silent drift from typos).

#### Scenario: Selecting the trainer variant with typed `stage2_ab`
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** a top-level `stage2_ab` mapping is provided
- **WHEN** training starts
- **THEN** the Stage-2 AB trainer is constructed and used for training
- **AND** the trainer reads Stage-2 AB knobs from `stage2_ab`.

#### Scenario: Unknown stage2_ab keys fail fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** a top-level `stage2_ab` mapping contains an unknown key (e.g., a typo)
- **WHEN** training starts
- **THEN** configuration parsing fails fast with guidance to fix/remove the unknown key.

### Requirement: Stage-2 AB remains compatible with ms-swift and Transformers (no upstream patches)
Stage-2 AB training MUST be implemented in a way that is compatible with:
- the ms-swift training harness (trainer construction, dataloader/collator semantics, and checkpoint/resume flow), and
- HuggingFace Transformers model forward contracts (no monkey-patching or editing upstream model files).

Normative constraints:
- The implementation MUST rely on existing ms-swift/Transformers mechanisms for loading Stage-1 pretrained checkpoints and resuming from checkpoints; it MUST NOT introduce a custom weight-loading pipeline.
- The trainer MUST preserve access to raw sample fields required by Channel-B (e.g., `messages`, `assistant_payload`) via an ms-swift compatible data-collation path (e.g., identity collator + `remove_unused_columns=False`), so rollout→parse→match construction can run.
- The trainer MUST filter out ms-swift / Trainer-injected helper keys before every model forward in both channels, and MUST NOT pass unknown/non-forward kwargs into the model.
  - At minimum, keys like `labels`, `compute_loss_func`, `loss_scale`, `text_position_ids`, and `channel` MUST NOT reach `model(...)` in either channel.
- The trainer MUST ensure that `outputs.logits` contains the full sequence-length logits needed for CE and coord decoding; it MUST NOT enable or rely on any logits-slicing mechanisms (e.g., `logits_to_keep`) that would drop positions required by the loss. If such a knob is configured or injected by the harness, the trainer MUST disable it (or fail fast with actionable guidance).
- When `training.packing` is enabled for Stage-2 AB, dataset-level packing MUST be disabled; Channel-B MUST use dynamic post-rollout packing for the teacher-forced forward pass (consistent with rollout-matching semantics).
  - When `training.packing` is enabled, Stage-2 AB MUST ensure **both** channels run with the padding-free packing metadata needed for Qwen3-VL correctness (i.e., `text_position_ids` + mRoPE `position_ids`) so each forward can pass the required 4-row `position_ids` format.
  - If the required packing metadata cannot be produced, the trainer MUST fail fast during initialization with actionable guidance (e.g., disable `training.packing`).

#### Scenario: Stage-2 AB runs under ms-swift with raw-sample collation
- **GIVEN** a config that selects `custom.trainer_variant: stage2_ab_training`
- **WHEN** training starts under ms-swift
- **THEN** the trainer receives raw samples with fields required for Channel-B (including `messages` and `assistant_payload`)
- **AND** no upstream Transformers files are modified to enable the run.

### Requirement: Channel selection is deterministic and step-driven
The trainer SHALL choose between Channel-A and Channel-B **deterministically** as a function of (`global_step`, `stage2_ab.schedule.b_ratio`).

Definition (normative):
- `global_step` MUST refer to the **optimizer-step** counter (post gradient-accumulation), i.e. the value that increments exactly once per optimizer update.
- The selected channel for a given `global_step` MUST remain fixed for the entire accumulation window (all micro-batches that contribute to that optimizer update).
- On resume from checkpoint, the schedule MUST continue from the restored `global_step` (no re-randomization).

Schedule definition (normative minimum):
- `stage2_ab.schedule.b_ratio` MUST be a float in `[0.0, 1.0]`.
- `stage2_ab.schedule.b_ratio` MUST be explicitly provided (no implicit default).
- Let optimizer step be `s` (0-indexed).
- The trainer MUST select Channel-B at step `s` iff:
  - `floor((s+1) * b_ratio) > floor(s * b_ratio)`.
  - Otherwise it MUST select Channel-A.

Special cases:
- If `b_ratio == 0.0`, the trainer MUST always select Channel-A.
- If `b_ratio == 1.0`, the trainer MUST always select Channel-B.

Legacy schedule handling (normative):
- The legacy list-based schedule knob `schedule.pattern` is not supported.
- If a config provides `stage2_ab.schedule.pattern`, configuration parsing MUST fail fast with guidance to migrate to `stage2_ab.schedule.b_ratio`.

Legacy rollout-buffer behavior:
- `custom.extra.rollout_matching.rollout_buffer` is removed.
- Any config that provides `custom.extra.rollout_matching.rollout_buffer` MUST fail fast with actionable guidance.

#### Scenario: b_ratio=0.5 alternates deterministically by optimizer step
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.5`
- **WHEN** the trainer selects channels for `global_step` `s = 0, 1, 2, 3`
- **THEN** it selects Channel-A at steps `0` and `2`
- **AND** it selects Channel-B at steps `1` and `3`.

#### Scenario: b_ratio edge cases are deterministic
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-A.

- **GIVEN** `stage2_ab.schedule.b_ratio: 1.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-B.

#### Scenario: Channel selection continues across checkpoint resume
- **GIVEN** a run that has completed optimizer step `global_step = s`
- **WHEN** training resumes from a checkpoint that restores `global_step = s`
- **THEN** the channel selected for step `s` is identical to the pre-resume selection for step `s`.

#### Scenario: stage2_ab.schedule.pattern fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** `stage2_ab.schedule.pattern: ["A","B"]` is provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to use `stage2_ab.schedule.b_ratio`.

#### Scenario: Missing b_ratio fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** `stage2_ab.schedule.b_ratio` is not provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to set `stage2_ab.schedule.b_ratio`.

#### Scenario: Legacy rollout_buffer config fails fast
- **GIVEN** a Stage-2 AB config that provides `custom.extra.rollout_matching.rollout_buffer`
- **WHEN** configuration is parsed/materialized
- **THEN** it fails fast with guidance to remove `rollout_buffer`.

### Requirement: Bbox-only v1 guardrails are enforced
The Stage-2 AB trainer MUST enforce bbox-only v1 guardrails on both GT objects and predicted rollout objects.

Stage-2 AB is bbox-only v1:
- GT objects MUST contain exactly one geometry field and it MUST be `bbox_2d` with exactly 4 coordinates.
  - If any GT object contains `poly` or any other geometry key, the trainer MUST fail fast with an error.
  - If any GT object contains malformed `bbox_2d` (wrong length, non-decodable values, or out-of-range values outside `[0, 999]` in norm1000 space), the trainer MUST fail fast.
    - Coercion contract (normative): values MUST be parsed as `int(round(float(x)))`; any value that cannot be coerced or lands outside `[0, 999]` is invalid.
  - If any GT bbox is invalid in ordering (`x2 < x1` or `y2 < y1`), the trainer MUST fail fast.

Predicted rollout objects MUST be filtered deterministically (no repair, no conversion):
- Instances whose geometry is not `bbox_2d` (including `poly`) MUST be dropped.
- Instances with missing/empty `desc` MUST be dropped.
- Instances with invalid bbox arity or invalid coord values MUST be dropped.

Diagnostics (normative):
- The trainer MUST expose strict-drop counters:
  - `N_valid_pred` (valid predicted instances kept),
  - `N_drop_invalid` (instances dropped by strict validation),
  - and reason buckets at minimum including: `missing_desc`, `missing_geom`, `wrong_arity`, `non_coord_token`, `poly_unsupported`, `unknown_geom`, `bbox_invalid`, `key_invalid`.
- The trainer MUST emit these counters as stable training metrics/log keys at least once per optimizer step that runs Channel-B.

#### Scenario: Drop diagnostics include valid and invalid counts
- **GIVEN** a Channel-B rollout containing a mix of valid bbox instances and invalid instances
- **WHEN** the trainer applies strict validation
- **THEN** invalid instances are dropped deterministically (no repair)
- **AND** the trainer exposes `N_valid_pred` and `N_drop_invalid` with at least one reason bucket incremented.

### Requirement: Channel-A performs iterative soft self-context via N× full-forwards (no rollout)
Channel-A MUST implement iterative soft self-context using `stage2_ab.n_softctx_iter` full forward passes:
- `stage2_ab.n_softctx_iter` MUST be an integer `>= 1`.
- The iteration index `m` ranges over `m = 0..n_softctx_iter-1`.
- For `n_softctx_iter = 1`, Channel-A MUST reduce to a single teacher-forced forward (pure TF baseline).
- For `n_softctx_iter > 1`, Channel-A MUST:
  - Run a teacher-forced forward to obtain logits for coord slots.
  - Construct coord-slot **soft embeddings** as the expectation over the coord-token sub-vocabulary.
  - Update coord-slot embeddings and re-run a full forward, repeating until `m = n_softctx_iter-1`.

The trainer MUST use the **final-iteration** logits `z^(n_softctx_iter-1)` for geometry decoding and loss computation.

Gradient semantics (normative):
- The default gradient mode MUST be fully unrolled:
  - `stage2_ab.softctx_grad_mode: "unroll"` MUST record gradients through all softctx iterations
  - and MUST NOT detach the expected coord embeddings used to update coord-slot inputs.
- An explicit EM-style fallback MAY be provided for ablations:
  - `stage2_ab.softctx_grad_mode: "em_detach"` MAY detach the expected coord embeddings (or equivalently disable grad recording for embedding updates) to reduce memory.

Causal shift convention (normative):
- For a causal LM, logits at sequence position `t` predict the token at position `t+1` (the standard shift used for CE).
- Therefore, when updating a coord-slot embedding at token position `p`, the trainer MUST use the coord distribution read from logits at position `p-1` (for `p > 0`) from the previous iteration.

#### Scenario: Unroll mode does not detach expected coord embeddings
- **GIVEN** `stage2_ab.n_softctx_iter: 2`
- **AND** `stage2_ab.softctx_grad_mode: unroll`
- **WHEN** Channel-A runs the softctx loop
- **THEN** it executes two full forward passes
- **AND** it does not detach the expected coord embeddings used to update coord-slot inputs.

### Requirement: Channel-A forward path is compatible with Qwen3-VL multimodal semantics
For Qwen3-VL (dense) models, each forward MUST provide **exactly one** of `input_ids` or `inputs_embeds`.

When Channel-A uses `inputs_embeds` to implement iterative soft self-context:
- The initial embeddings MUST be computed by **calling** the model's input embedding module on the teacher-forced token ids (so embedding forward hooks apply); they MUST NOT be assembled by indexing `.weight[...]`.
- Expected coord embeddings used in soft self-context MUST also be computed via the embedding module applied to coord token ids (not via `.weight[...]`).
- The trainer MUST modify only the coord-slot embeddings and MUST NOT perturb multimodal placeholder token embeddings (e.g., image/video placeholders) **bitwise**.
- For multimodal batches, each softctx iteration MUST provide `inputs_embeds` that still contains placeholder token embeddings at placeholder positions (so Qwen3-VL can insert visual features).
  - Normative minimum: each iteration MUST rebuild a fresh base `inputs_embeds` from teacher-forced `input_ids` (embedding-module call) and then scatter-update only coord-slot rows.
  - It MUST NOT reuse embeddings after model-internal feature insertion for subsequent iterations.
- Channel-A MAY retain `input_ids` in the batch for label/slot bookkeeping, but MUST NOT pass `input_ids` into `model(...)` when `inputs_embeds` is provided.
- The forward call MUST remain compatible with multimodal feature insertion (i.e., placeholder token count must match provided visual feature length).
- When using padding-free packing with Qwen3-VL, the trainer MUST pass the 4-row `position_ids` format (`[text_position_ids; mRoPE]`) consistently for every iteration forward.
- Cache safety (normative): Channel-A training forwards MUST set `use_cache=False` and MUST NOT pass or reuse `past_key_values` across iterations. Carrying KV cache across softctx iterations is forbidden.

#### Scenario: Multimodal forward does not fail under iterative inputs_embeds
- **GIVEN** a multimodal batch with `pixel_values` and valid placeholder tokens
- **AND** Channel-A runs with `n_softctx_iter > 1`
- **WHEN** the model is forwarded with `inputs_embeds` (and `input_ids=None`)
- **THEN** the forward pass succeeds without placeholder-count mismatch errors.

### Requirement: Channel-B reuses rollout-matching infra (strict parse/match + mandatory FN append)
Channel-B MUST reuse the rollout-matching pipeline:
- Rollout generation MUST be configured under `custom.extra.rollout_matching` (backend `hf` or `vllm`).
- Parsing MUST be strict and token-aligned (no re-tokenization of the rollout prefix), except for a possible token-internal cut on the final token where the trainer MAY retokenize only the final token as a shorter tokenization that decodes exactly to the original substring.
- Matching MUST be deterministic.
- FN append MUST be performed (mandatory) to ensure all GT objects are present in `Y_train`.

#### Scenario: Rollout prefix + FN append produces a valid teacher-forced target
- **GIVEN** Channel-B is selected and rollout generation succeeds
- **WHEN** the trainer builds `Y_train` for teacher forcing
- **THEN** `Y_train` contains the rollout prefix (suffix-trimmed only) followed by a JSON-only FN append fragment.

### Requirement: Channel-B invalid rollouts fall back deterministically (no silent skips)
When Channel-B is selected and a rollout response cannot be parsed into an append-ready JSON prefix (e.g., there is no top-level `{` in the response), the trainer MUST:
- Mark the rollout as invalid for that sample and expose an `invalid_rollout` counter/metric.
- Fall back to a canonical empty JSON prefix of exactly `{` (as token ids) so FN append can proceed.
- Treat the rollout as containing zero valid predicted objects for matching/supervision purposes.
- Continue training that sample by FN-appending all GT objects and running the normal teacher-forced loss (i.e., the sample is not skipped and the trainer does not raise).

This fallback MUST be deterministic given the same `response_token_ids` and tokenizer.

Normative minimum: this requirement MUST at least cover the case where the rollout response contains no top-level `{` (equivalent to the current strict parser’s “completely malformed rollout” condition).

#### Scenario: Missing opening brace falls back to `{` and trains
- **GIVEN** Channel-B is selected for a sample
- **AND** the rollout response text contains no top-level `{`
- **WHEN** the trainer parses the rollout for matching
- **THEN** it marks the rollout invalid for that sample
- **AND** it uses `{` as the prefix and FN-appends all GT objects
- **AND** the sample is still included in teacher-forced training.

### Requirement: Channel-B rollout seeding is deterministic and logged
Channel-B rollouts MUST be fully deterministic under greedy decoding given the same model weights, the same training seed, and the same `global_step`.

If stochastic decoding is enabled, the trainer MUST apply a deterministic seeding plan and SHOULD be reproducible under a fixed run configuration (same backend, same world size/sharding, same batch shapes/order). Exact per-request determinism for stochastic decoding is not required in this capability.

Definition (normative):
- `training_seed` refers to HuggingFace `TrainingArguments.seed` (ms-swift `train_args.seed`), i.e., the same seed source used by `rollout_matching_sft`.

Deterministic decoding constraints (normative):
- The trainer MUST support deterministic greedy decoding for Channel-B rollouts under both HF and vLLM backends.
- If the run enables stochastic decoding (e.g., `temperature > 0` and/or `do_sample: true`), the trainer MUST make it reproducible via deterministic seeding derived from `rollout_seed_base`.
  - For `rollout_backend: "vllm"`, seeded stochastic decoding MUST be implemented by passing per-request seeds via the backend request config (or equivalent).
  - For `rollout_backend: "hf"`, if stochastic decoding is enabled, the trainer MUST at least set a deterministic *global* RNG seed before rollout generation (e.g., `seed_everything(rollout_seed_base)`), and MUST log `rollout_seed_base` and the decoding params used.
    - Per-request RNG isolation is OPTIONAL (and may be added later), but is not required for landing this capability.

The trainer MUST derive a deterministic rollout seed base consistent with the rollout-matching seeding contract:
- `rollout_seed_base = (training_seed + global_step * 1000003) & 0x7FFFFFFF`

Per-request seeds MUST be derived deterministically from `rollout_seed_base` and the within-batch request index (or an equivalent stable index plan for multi-server chunking).

The trainer MUST log (or expose via metrics) the effective `rollout_seed_base` at least once per optimizer step when Channel-B performs fresh rollouts.

#### Scenario: Same step produces identical derived rollout seed base
- **GIVEN** `training_seed = 123` and `global_step = 7`
- **WHEN** Channel-B derives the rollout seed base twice
- **THEN** both runs produce the same `rollout_seed_base`
- **AND** the derived `rollout_seed_base` is logged for reproducibility.

### Requirement: Hybrid objective preserves JSON structure CE and adds bbox geometry losses
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- **Token CE anchor at A1**:
  - CE on non-coord tokens MUST be computed from the teacher-forced logits of the first forward (`z^(0)`; GT context).
  - Coord tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Geometry + distribution regularizers from final softctx logits**:
  - Geometry losses and any distribution-level losses MUST be computed from the final-iteration logits `z^(n_softctx_iter-1)`.

Channel-B:
- **Matched-only geometry**:
  - Geometry losses MUST be computed only for matched `(pred_i -> gt_j)` pairs.
  - Unmatched predicted objects (FP under Hungarian) MUST NOT receive geometric gradients.
- **Stop-neutral CE**:
  - Channel-B CE MUST NOT supervise the stop/continue decision.
  - The trainer MUST mask CE on:
    - the top-level JSON closing brace `}` (the brace that closes the outermost assistant JSON object), and
    - `<|im_end|>` (the only turn-end token).
  - Top-level brace identification MUST be robust and deterministic:
    - the trainer MUST identify the token position of the `}` that closes the **outermost** JSON object in the rendered `y_GT_reordered` assistant span,
    - and MUST NOT rely on “the last `}` token id in the whole sequence” without verifying it corresponds to the outermost close brace of the assistant JSON.
    - A compliant approach is to decode the assistant-span token pieces and locate the outermost close brace via a brace-depth scan, then map the character span back to token positions.
- **FN append always**:
  - FN objects MUST be appended to the B3 target so they are supervised even when they were missing from rollout.
  - If `N_valid_pred == 0` after strict validation, the trainer MUST fall back to `y_GT_reordered := y_GT_canonical` (Stage-1 canonical GT order), which is equivalent to “FN append all GT objects”.
  - Optional weak correction: when `N_drop_invalid > 0`, the trainer MAY upweight Channel-B’s B3 structure-token CE weights to discourage “escaping supervision via invalid instances”.
    - This upweight MUST be controlled by `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` (float).
    - The multiplier MUST default to `1.0` (no effect) and MUST be constrained to a safe range `[1.0, 4.0]` (clamp or fail fast).
    - “Structure-token CE weights” refers to Channel-B CE-supervised tokens excluding:
      - coord tokens,
      - desc value tokens, and
      - stop-neutral masked token positions (`}` and `<|im_end|>`).

Configurable desc supervision (both channels):
- `stage2_ab.desc_ce_weight` MUST be a float `>= 0` and applies to desc value tokens by default.
- Channel-B MAY additionally provide `stage2_ab.channel_b.desc_ce_weight_matched` as the weight for matched-object desc tokens (distinct from FN-appended desc tokens).

Bbox geometry losses (both channels) are computed from coord distributions:
- The trainer MUST decode coordinates from coord-token distributions via CoordExp expectation decoding (not argmax):
  - Let bins `k ∈ {0..999}` correspond to the coord-token sub-vocabulary.
  - Let `p(k)` be the softmax distribution over these 1000 bins for a coord slot, taken from the standard causal shift:
    - For a coord token at input position `p`, `p(k)` MUST be computed from logits at position `p-1` (consistent with CE).
  - The decoded normalized coordinate MUST be: `c_hat = Σ_k p(k) * (k/999)` in `[0, 1]`.
- The trainer MUST compute bbox losses in normalized coordinate space `[0, 1]`:
  - GT bbox ints in `[0, 999]` MUST be converted to floats by dividing by `999`.
  - Predicted bbox coords MUST be the decoded normalized floats from `c_hat` above.
- Geometry loss MUST use logits from:
  - the final iteration `z^(n_softctx_iter-1)` in Channel-A, and
  - the Channel-B teacher-forced logits under the rollout scaffold (or B2 refined logits when enabled).

Loss form (normative):
- The trainer MUST use SmoothL1 (Huber) + CIoU as the bbox regression terms.
- The trainer MUST NOT use GIoU in Stage-2 AB.

Numerical stability (normative):
- The trainer MUST canonicalize predicted boxes before CIoU:
  - `(x1,x2) := (min(x1,x2), max(x1,x2))`, `(y1,y2) := (min(y1,y2), max(y1,y2))`.
- The trainer MUST ensure the geometry losses do not produce NaNs/Infs, including early training when predictions are degenerate.

Efficiency rule (normative):
- If Channel-B has no valid matched pairs for a sample/batch, the trainer MUST skip the B2 forward (geo-only) and run B3 only.

#### Scenario: Channel-B is stop-neutral for `}` and `<|im_end|>`
- **GIVEN** Channel-B builds a teacher-forced target that ends with a top-level `}` followed by `<|im_end|>`
- **WHEN** the trainer builds CE labels/weights for Channel-B
- **THEN** it masks CE on that top-level `}` token position
- **AND** it masks CE on `<|im_end|>`.

#### Scenario: Dropped invalid instances may upweight B3 structure CE
- **GIVEN** a Channel-B rollout with `N_drop_invalid > 0`
- **AND** `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier: 1.5`
- **WHEN** Channel-B builds CE weights for structure tokens in B3
- **THEN** it MAY multiply the structure-token CE weights by `1.5` (bounded) for that sample/window.

### Requirement: Coord quantization is globally consistent (k/999 only)
The Stage-2 AB trainer MUST use a single consistent coord quantization scheme:
- Encode: `k = clamp(round(999*c), 0, 999)`
- Decode: `c = k/999`

Semantics (normative):
- Coord tokens of the form `<|coord_k|>` denote an integer bin index `k ∈ [0, 999]` under the project’s 1000-bin convention (bins 0..999).
- In this convention, `(k_x, k_y) = (0, 0)` corresponds to the top-left pixel corner, and `(999, 999)` corresponds to the bottom-right pixel corner (i.e., the inclusive image bounds).
  - No `/1000`-based "bin -> normalized float" mapping is allowed anywhere in the repository. Any existing `/1000`-based helpers/metadata MUST be removed or updated to `/999` so that `k=999` maps to `1.0` consistently.

The trainer MUST NOT use any `1000`-based normalization (e.g., `round(1000*c)` or decode `k/1000`).

#### Scenario: Upper bound is 999 and decodes to 1.0
- **GIVEN** a normalized coordinate `c = 1.0`
- **WHEN** it is encoded to a bin index
- **THEN** the bin index is `k = 999`
- **AND** decoding returns `k/999 = 1.0`
- **AND** no bin index `1000` is ever produced.

#### Scenario: Lower bound is 0 and decodes to 0.0
- **GIVEN** a normalized coordinate `c = 0.0`
- **WHEN** it is encoded to a bin index
- **THEN** the bin index is `k = 0`
- **AND** decoding returns `k/999 = 0.0`.

### Requirement: Channel-B step mode is step-budgeted in raw rollouts and learns-to-completion under packing
When Stage-2 AB training is enabled and Channel-B executes in step mode (`custom.extra.stage2_ab.channel_b.mode=step`), the trainer SHALL interpret the Channel-B batch size in terms of **raw rollouts per optimizer step**, not “packed sequences per optimizer step”.

Normative behavior:
- The trainer MUST collect `rollouts_per_step` raw rollouts **globally across all train ranks** for the optimizer step.
  - Under DDP, each rank MUST collect a deterministic share `local_rollouts_per_step` such that the sum over ranks equals `rollouts_per_step`.
  - If `custom.extra.stage2_ab.channel_b.rollouts_per_step` is unset, the trainer MUST default it to the **derived global effective batch size**:
    - `training.per_device_train_batch_size × world_size × training.gradient_accumulation_steps`
    - Note: when using ms-swift `training.effective_batch_size`, `training.gradient_accumulation_steps` is auto-derived (ceil), so the derived global effective batch size MAY be >= the user-requested value.
- The trainer MUST then construct per-sample teacher-forced segments (rollout prefix + mandatory FN append).
- When `training.packing=true`, the trainer MUST pack these segments into a **variable** number of packed sequences under the `packing_length` cap derived from `global_max_length`.
- The trainer MUST run forward/backward once per packed sequence and accumulate gradients, then perform **exactly one** optimizer update for the optimizer step.

#### Scenario: 32 raw rollouts pack into fewer than 32 packed sequences
- **GIVEN** `training.effective_batch_size=32`
- **AND** `training.packing=true` and `global_max_length=12000`
- **WHEN** Channel-B executes for one optimizer step
- **THEN** the trainer collects 32 raw rollouts
- **AND** packs them into `N_packs` packed sequences where `N_packs` MAY be less than 32
- **AND** performs one optimizer update for the step.

#### Scenario: Channel-B executes only on the final micro-step under grad accumulation
- **GIVEN** `training.per_device_train_batch_size=1`, `world_size=4`, and `training.gradient_accumulation_steps=8`
- **AND** `custom.extra.stage2_ab.channel_b.mode=step`
- **WHEN** Channel-B is selected for one optimizer step
- **THEN** each rank buffers its raw rollouts across the first 7 micro-steps without running the Channel-B loop
- **AND** the full Channel-B loop (rollout→pack→learn-to-completion) runs on the 8th (final) micro-step
- **AND** the outer Trainer performs exactly one optimizer update for the step.

### Requirement: Channel-B step mode supports an in-step bounded pipeline queue between rollout and learning
When Channel-B executes in step mode with packing enabled, the trainer SHALL support overlapping rollout generation with learner compute within the optimizer step using a bounded producer/consumer queue (size 1 is sufficient).

Normative safety guardrail:
- If the in-step pipeline queue is enabled, rollouts MUST run on dedicated GPUs via vLLM server mode.
  - Concretely: the trainer MUST require `custom.extra.rollout_matching.rollout_backend=vllm` and `custom.extra.rollout_matching.vllm.mode=server`.
  - If this condition is not met, the trainer MUST error fast with a clear message (to avoid unsafe concurrent rollout+train on the same process/device).

#### Scenario: Rollout and learner overlap within a step
- **GIVEN** rollout runs on dedicated GPUs via vLLM server mode
- **AND** learner training runs on a separate GPU
- **WHEN** Channel-B executes one optimizer step
- **THEN** the trainer overlaps rollout generation and learner forward/backward where feasible
- **AND** the trainer does not build an unbounded rollout pool.

### Requirement: Channel-B rollout decode batching is configurable and independent of learner microbatch
When Channel-B executes, the trainer SHALL allow configuring rollout decode batching (e.g., 2) independently of learner microbatch size (which remains 1 under packing).

#### Scenario: Rollout decode batch size 2 with learner microbatch 1
- **GIVEN** `training.per_device_train_batch_size=1`
- **AND** Channel-B rollout decode batch size is configured as 2
- **WHEN** rollouts are generated
- **THEN** the rollout backend generates responses for 2 samples in one decode call
- **AND** learner training still runs one packed sequence per forward/backward.

### Requirement: Channel-B supports semantic-tolerant matched desc supervision
Channel-B desc supervision MUST support a semantic tolerance mode for matched objects:
- For each matched pair `(pred_i -> gt_j)`, compute a similarity score between:
  - the predicted description string `pred_desc_i` from rollout parsing, and
  - the GT description string `gt_desc_j` from the dataset.
- If similarity is at least a configurable threshold, the trainer MUST treat the predicted desc as acceptable and MUST NOT penalize the matched object’s GT desc token positions in CE (i.e., weight 0 / masked).
- If similarity is below the threshold, the trainer MUST apply a (small) desc CE weight to pull toward GT.

Scope constraints (normative):
- Semantic tolerance MUST apply only to **matched** objects.
- FN-appended objects MUST be supervised normally (no semantic gating), since they have no competing predicted description.

Configuration (normative):
- `stage2_ab.channel_b.semantic_desc_gate.enabled` MUST accept a boolean and MUST default to `true`.
- `stage2_ab.channel_b.semantic_desc_gate.threshold` MUST accept a float in `[0.0, 1.0]` and MUST default to `0.5`.
- `stage2_ab.channel_b.semantic_desc_gate.model_name_or_path` MUST accept a string and MUST default to `sentence-transformers/all-MiniLM-L6-v2`.
- If semantic gating is enabled, `stage2_ab.channel_b.semantic_desc_gate.revision` MUST be provided as a string to pin the embedding model version.
- If semantic gating is enabled, the resolved model identity MUST be logged for reproducibility (including the provided `revision`).
- If semantic gating is enabled but the sentence-transformer dependency or the specified model weights are not available at runtime, the trainer MUST NOT fail fast and MUST instead:
  - disable semantic gating for the affected step/run (treating matched desc tokens as not semantically acceptable unless they match the normal non-gated rules), and
  - emit a stable warning log at least once describing that semantic gating is disabled due to missing dependency/weights, and
  - expose a stable boolean metric/log key indicating whether semantic gating is active for the step (e.g., `stage2_ab/channel_b/semantic_desc_gate/is_active`).
Performance (non-normative guidance):
- The implementation SHOULD compute sentence embeddings in a batched manner per optimizer step (or per micro-batch) and MAY cache embeddings within the step to bound overhead without changing semantics.

#### Scenario: Semantically close matched desc is not penalized
- **GIVEN** a matched object whose rollout desc is semantically close to GT (similarity ≥ threshold)
- **WHEN** Channel-B builds CE labels/weights for the matched object’s GT desc tokens
- **THEN** those desc token positions are masked (weight 0) so the model is not forced to match the GT string exactly.

#### Scenario: Semantic gate is disabled when the model is unavailable
- **GIVEN** `stage2_ab.channel_b.semantic_desc_gate.enabled: true`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.model_name_or_path: "/path/does/not/exist"`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.revision: "pinned"`
- **WHEN** training starts and Channel-B attempts to compute semantic gating
- **THEN** the trainer continues without semantic gating for that step/run
- **AND** it emits a warning that semantic gating is disabled due to missing dependency/weights
- **AND** it exposes `stage2_ab/channel_b/semantic_desc_gate/is_active = false`.

#### Scenario: Semantic gate requires a pinned revision/version
- **GIVEN** `stage2_ab.channel_b.semantic_desc_gate.enabled: true`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.revision` is not provided
- **WHEN** training starts and Channel-B attempts to load the semantic gate model
- **THEN** the trainer fails fast with guidance to provide `stage2_ab.channel_b.semantic_desc_gate.revision`.

### Requirement: Deprecated legacy coord-loss knobs are silently ignored
To enable config refactors without blocking training runs, the configuration system MUST silently ignore deprecated legacy coord-loss knobs under `custom.*` that are no longer supported by the project’s coord-loss contract.

Normative minimum:
- If `custom.coord_loss` is present in a YAML config, configuration parsing MUST NOT raise, and the value MUST be ignored.

#### Scenario: custom.coord_loss does not hard error
- **GIVEN** a config that includes `custom.coord_loss` (legacy)
- **WHEN** configuration is parsed
- **THEN** parsing succeeds and the legacy field is ignored.
