# stage2-ab-training Specification

## Purpose
Provide Stage-2 two-channel training infrastructure for CoordExp that combines:
- **Channel-A (hot)**: iterative soft self-context via `n_softctx_iter` full-forwards (no autoregressive rollout).
- **Channel-B (cold)**: deterministic self-rollout + strict parse/match + teacher-forced learning.

This capability is intended to fine-tune a model that has already been trained in Stage-1 (teacher-forced pretraining). As a result, rollouts are expected to be relatively stable/syntactically well-formed (though they may still be inaccurate or miss objects). The pipeline MUST still define deterministic behavior for invalid rollouts.

This capability is **bbox-only v1**: GT polygons are rejected (fail fast). The implementation MUST remain scalable to polygon rollouts/losses in a future change.

## ADDED Requirements
### Requirement: Stage-2 AB trainer variant is selectable via YAML
When training config sets `custom.trainer_variant: stage2_ab_training`, the system SHALL use the Stage-2 AB trainer implementation.

The trainer MUST be configurable via YAML under `custom.extra.stage2_ab` and MUST NOT require new CLI flags.

#### Scenario: Selecting the trainer variant
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **WHEN** training starts
- **THEN** the Stage-2 AB trainer is constructed and used for training.

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
The trainer SHALL choose between Channel-A and Channel-B **deterministically** as a function of (`global_step`, `schedule.pattern`), with a deterministic override to Channel-B when rollout-buffer reuse is active.

Definition (normative):
- `global_step` MUST refer to the **optimizer-step** counter (post gradient-accumulation), i.e. the value that increments exactly once per optimizer update.
- The selected channel for a given `global_step` MUST remain fixed for the entire accumulation window (all micro-batches that contribute to that optimizer update).
- On resume from checkpoint, the schedule MUST continue from the restored `global_step` (no re-randomization).

The schedule MUST support a simple repeating pattern (normative minimum):
- `schedule.pattern`: a non-empty list of strings in `{"A","B"}`.
- At optimizer step `global_step = s`, the chosen channel MUST be `schedule.pattern[s % len(pattern)]`.

If rollout buffering is enabled and the trainer is reusing a buffered Channel-B batch for M-steps, the trainer MUST keep using Channel-B for those reuse steps (i.e., buffer reuse deterministically overrides the schedule) to preserve deterministic semantics.

Buffering determinism (normative):
- If buffering is enabled with `custom.extra.rollout_matching.rollout_buffer.m_steps > 1`, the trainer MUST disable reuse deterministically on any partial final accumulation window (i.e., fewer than `gradient_accumulation_steps` micro-batches).
- The run SHOULD set `training_args.dataloader_drop_last=true` (or equivalent) to avoid partial final accumulation windows; however, determinism MUST hold even if drop_last is false (by disabling reuse on the partial window).
- If the trainer cannot ensure safe/deterministic reuse under the current dataloader/accumulation configuration, it MUST force `m_steps=1` for that run (or fail fast with actionable guidance).

#### Scenario: Pattern schedule repeats deterministically
- **GIVEN** `schedule.pattern: ["A","A","B"]`
- **WHEN** `global_step` is 0, 1, 2, 3, 4
- **THEN** the selected channels are A, A, B, A, A respectively.

### Requirement: Bbox-only v1 guardrails are enforced
In Stage-2 AB training, the system MUST enforce bbox-only v1 rules:
- GT objects MUST contain exactly one geometry field and it MUST be `bbox_2d` with exactly 4 coordinates.
  - If any GT object contains `poly` or any other geometry key, the trainer MUST fail fast with an error.
  - If any GT object contains malformed `bbox_2d` (wrong length, non-decodable values, or out-of-range values outside `[0, 999]` in norm1000 space), the trainer MUST fail fast.
    - Coercion contract (normative): values MUST be parsed as `int(round(float(x)))`; any value that cannot be coerced or lands outside `[0, 999]` is invalid.
  - If any GT bbox is invalid in ordering (`x2 < x1` or `y2 < y1`), the trainer MUST fail fast.
- Predicted rollout objects whose geometry is not `bbox_2d` (including `poly`) MUST be dropped deterministically (no repair, no conversion).
- Predicted rollout objects with `bbox_2d` but an invalid coord count or invalid coord values MUST be dropped deterministically.
- The trainer MUST expose drop counters (at least by geometry kind: `poly`, `unknown`, `bbox_invalid`).

#### Scenario: GT poly fails fast
- **GIVEN** a training sample whose GT `assistant_payload` contains an object with `poly`
- **WHEN** the trainer prepares the sample for either channel
- **THEN** it raises an error indicating bbox-only v1 requires filtering out polygons upstream.

### Requirement: Channel-A performs iterative soft self-context via N× full-forwards (no rollout)
Channel-A MUST implement iterative soft self-context using `n_softctx_iter` full forward passes:
- `custom.extra.stage2_ab.n_softctx_iter` MUST be an integer `>= 1`.
- The iteration index `m` ranges over `m = 0..n_softctx_iter-1`.
- For `n_softctx_iter = 1`, Channel-A MUST reduce to a single teacher-forced forward (pure TF baseline).
- For `n_softctx_iter > 1`, Channel-A MUST:
  - Run a teacher-forced forward to obtain logits for coord slots.
  - Construct coord-slot **soft embeddings** as the expectation over the coord-token sub-vocabulary.
  - Update coord-slot embeddings and re-run a full forward, repeating until `m = n_softctx_iter-1`.

The trainer MUST use the **final-iteration** logits `z^(n_softctx_iter-1)` for geometry decoding and loss computation (intermediate-iteration losses MAY exist but are not required).

For memory stability, iterations `m = 0..n_softctx_iter-2` MUST run under `no_grad` (default), and only the final iteration MUST run with grad.
The trainer MUST NOT change the model's train/eval mode inside the softctx loop; only gradient recording may be disabled for early iterations.

Causal shift convention (normative):
- For a causal LM, logits at sequence position `t` predict the token at position `t+1` (the standard shift used for CE).
- Therefore, when updating a coord-slot embedding at token position `p`, the trainer MUST use the coord distribution read from logits at position `p-1` (for `p > 0`) from the previous iteration.

#### Scenario: n_softctx_iter=2 runs exactly two forwards and uses final logits
- **GIVEN** `n_softctx_iter: 2`
- **WHEN** Channel-A runs on a batch
- **THEN** it executes two full forward passes
- **AND** it computes geometry loss using logits from the second forward only.

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
- **Token CE** on non-coord tokens that represent JSON structure (braces/commas/keys/etc.), consistent with existing tail CE semantics.
  - Coord tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Configurable desc supervision**:
  - `custom.extra.stage2_ab.desc_ce_weight` MUST be a float `>= 0`.
  - Desc *value* tokens MAY be down-weighted by `desc_ce_weight`.
  - When `desc_ce_weight == 0`, desc value tokens MUST be fully masked out from CE.
- **Bbox geometry losses** computed from coord distributions:
  - The trainer MUST decode coordinates from coord-token distributions via CoordExp expectation decoding (not argmax):
    - Let bins `k ∈ {0..999}` correspond to the coord-token sub-vocabulary.
    - Let `p(k)` be the softmax distribution over these 1000 bins for a coord slot, taken from the standard causal shift:
      - For a coord token at input position `p`, `p(k)` MUST be computed from logits at position `p-1` (consistent with CE).
      - Unless explicitly configured, `p(k)` MUST be computed with temperature `1.0` (standard softmax over coord logits; no implicit temperature).
    - The decoded normalized coordinate MUST be: `c_hat = E_p[k] / 999 = Σ_k p(k) * (k/999)` in `[0, 1]`.
  - The trainer MUST compute bbox L1 loss and generalized IoU loss in normalized coordinate space `[0, 1]`:
    - GT bbox ints in `[0, 999]` MUST be converted to floats by dividing by `999`.
    - Predicted bbox coords MUST be the decoded normalized floats from `c_hat` above.
    - The trainer MUST NOT compute these geometry losses in pixel space in this capability.
    - The trainer MUST NOT compute these geometry losses directly on raw integer bins without first converting to normalized floats (divide by `999`).
    - The trainer MUST ensure geometry losses are numerically stable even when the predicted bbox is non-canonical (e.g., `x1 > x2`) or partially out of range; it MUST NOT produce NaNs/Infs.
      - A compliant approach is to canonicalize predicted bboxes via `(x1,x2) = (min(x1,x2), max(x1,x2))`, `(y1,y2) = (min(y1,y2), max(y1,y2))`, then clip each coordinate to `[0, 1]` before computing GIoU.
  - Geometry loss MUST use logits from the final iteration: `z^(n_softctx_iter-1)` in Channel-A and the teacher-forced logits in Channel-B.

#### Scenario: Desc can be fully masked while keeping structure CE
- **GIVEN** `desc_ce_weight: 0`
- **WHEN** CE labels are built for a batch
- **THEN** JSON structure tokens remain supervised by CE
- **AND** desc value tokens do not contribute to CE.

#### Scenario: Expectation decoding uses probability-weighted mean (not argmax)
- **GIVEN** a coord-slot distribution with `p(k=0)=0.5` and `p(k=999)=0.5`
- **WHEN** the trainer decodes the coordinate via expectation decoding
- **THEN** the decoded value is approximately `0.5` (i.e., `(0*0.5 + 999*0.5)/999`)
- **AND** it is not equal to an argmax decode of `0` or `1`.

#### Scenario: Geometry losses operate on normalized coordinates
- **GIVEN** a GT bbox coordinate bin value `k=999`
- **WHEN** the trainer converts GT bins to normalized floats for geometry loss
- **THEN** the converted value is `999/999 = 1.0`
- **AND** geometry losses are computed using normalized floats in `[0, 1]`.

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
