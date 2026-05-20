---
doc_id: docs.training.stage1-objective
layer: docs
doc_type: reference
status: canonical
domain: training
summary: Stage-1 objective surfaces and coord-token training behavior.
updated: 2026-05-20
---

# Coord Objective & Adapter

This document details the specialized training objectives and architectural adapters used for coordinate tokens in CoordExp.

Scope note:
- This page is primarily the Stage-1 / baseline coord-objective reference.
- The canonical Stage-1 packing contract now lives in [`../data/PACKING.md`](../data/PACKING.md):
  one hard `global_max_length` cap, offline static packing, full-length probing before plan build,
  and fail-fast when any atomic sample exceeds the cap.
- For Stage-2 pipeline-declared training, the canonical objective surface now lives under:
  - `stage2_ab.pipeline` for `custom.trainer_variant: stage2_two_channel`
  - `rollout_matching.pipeline` for `custom.trainer_variant: stage2_rollout_aligned`
- In those Stage-2 paths, `coord_reg`, `bbox_geo`, and `loss_duplicate_burst_unlikelihood` are declared through the pipeline surface described in:
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
- Legacy `custom.coord_soft_ce_w1.*` authoring should not be used for pipeline-declared Stage-2 configs.
- For standard Stage-1 SFT, the active non-pipeline teacher-forcing surface is:
  - `custom.coord_soft_ce_w1.*`
  - `custom.bbox_geo.*`
  - `custom.bbox_size_aux.*`
- Raw-text norm1000 and standalone bbox-geometry profile YAMLs were removed
  from the current runnable config surface. Historical evidence remains in
  `progress/`, archived OpenSpec changes, and run artifacts.
- Compact prefix roll-in multi-positive training now lives as the detection `prefix_rollin_et_rmp_ce` ablation surface, not as a legacy custom trainer surface. The first checked-in route is `configs/stage1/recursive_detection_ce/ablation/compact_full_prefix_rollin_balance2.yaml`; treat it as E1 ablation/smoke validation, not production.
- Geometry-aware coordinate SoftCE for compact recursive detection is scoped to the retained focused cap8 instance-trie successors under `configs/stage1/recursive_detection_ce/prod/`. It uses `objective.coord_soft_ce` and does not route through legacy `custom.coord_soft_ce_w1.*`.
- Narrow V1 exception:
  - `custom.bbox_format: cxcy_logw_logh` or `custom.bbox_format: cxcywh`
    defines an experimental Stage-1-only profile
  - under that profile, the allowed Stage-1 surface narrows to
    `custom.coord_soft_ce_w1.*` with hard CE plus positive coord/text gating
  - `custom.bbox_geo.*`, `custom.bbox_size_aux.*`, soft CE, W1, and trainer-side
    rollout/Stage-2 surfaces are intentionally out of scope and should be
    treated as invalid for that experiment

## Current Mechanism Note (Interpretation, Not Stable Contract)

Inference-only duplication studies on existing `merged` checkpoints now support
a more specific rollout-risk framing than the earlier generic "attention drifts
away from vision" explanation:

- the strongest onset-local separator is the early coordinate escape behavior at
  `x1` and `y1`
- healthy same-desc continuations usually evacuate probability mass away from
  the previous or local bbox neighborhood quickly
- duplicated continuations often keep `x1` / `y1` diffuse, high-entropy, or
  locally sticky long enough for rollout history to lock the model into a
  repeated-object basin
- late history overwrite still matters, but current control evidence suggests
  it is better treated as a secondary amplifier than as the sole root cause

Working interpretation:

- `softCE`, `W1`, and expectation-decoded geometry can preserve smooth local
  coordinate structure that looks acceptable under teacher forcing
- during rollout, that same local smoothness can lower the escape barrier
  between nearby same-desc instances
- once the model fails to separate from the previous or local basin at
  `coord_x1` / `coord_y1`, prior generated coord tokens and recent history can
  make duplication self-reinforcing

This does **not** yet prove that clean from-scratch pure CE fully solves the
problem. The current CE-side references on disk remain continuation-style
proxies unless a token-compatible pure-CE checkpoint is evaluated under the
same onset-local protocol.

## Coord distribution loss (coord tokens)

CoordExp can supervise coordinate tokens with **distribution-based losses**
(recommended default for the existing `xyxy` Stage-1 baseline):

- Standard full-vocab CE is applied **only to non-coordinate tokens** (text + JSON structure).
- At `<|coord_*|>` positions, the model is supervised via:
  - `CE` (optional): hard CE over the 1000-bin coord vocabulary (ablation knob; default `0.0`)
  - `softCE`: soft cross-entropy between predicted coord-bin distribution `p` and a unimodal Gaussian soft label `q`
  - `W1`: 1D Wasserstein-1 distance on discrete bins via CDF differences between `p` and `q`
  - `gate`: coord-vocab gate loss that penalizes probability mass leaking to non-coord tokens

```yaml
custom:
  coord_soft_ce_w1:
    enabled: true
    # total_loss += ce_weight * CE + soft_ce_weight * softCE + w1_weight * W1
    #             + gate_weight * gate + adjacent_repulsion_weight * adjacent_repulsion
    ce_weight: 0.0
    soft_ce_weight: 1.0
    w1_weight: 1.0
    gate_weight: 1.0
    temperature: 1.0
    target_sigma: 2.0
    target_truncate: 16
    adjacent_repulsion_weight: 0.0
    adjacent_repulsion_filter_mode: same_desc
    adjacent_repulsion_margin_ratio: 0.05
    adjacent_repulsion_copy_margin: 0.8
```

**Notes**:
- Coord-token positions are identified from **labels** (teacher forcing), never from model predictions.
- No decoded coordinates (argmax/expectation/median) are computed for training or metrics.
- Because this objective is optimized under teacher forcing, it does not by
  itself test whether rollout can escape a previously emitted same-desc local
  basin. The active duplication-collapse analysis therefore treats early
  `coord_x1` / `coord_y1` escape from the previous/local neighborhood as the
  primary rollout diagnostic surface.
- Logged losses (train/eval parity, eval uses `eval_` prefix):
  - Stage-1 coord-family loss keys include `coord_softce_w1/loss`, `coord_softce_w1/soft_ce`, `coord_softce_w1/w1`, `coord_softce_w1/gate`, and `coord_softce_w1/adjacent_repulsion`
  - Stage-1 coord diagnostics include `coord_diag/loss`, `coord_diag/soft_ce`, `coord_diag/w1`, `coord_diag/gate`, `coord_diag/adjacent_repulsion`, `coord_diag/adjacent_repulsion_pair_count`, `coord_diag/adjacent_repulsion_applied_count`, `coord_diag/adjacent_repulsion_copy_score_mean`, plus `coord_diag/coord_vocab_mass`, `coord_diag/coord_tokens`, and the mode flag `coord_diag/enabled`
- Stage-2 note:
  - `stage2_two_channel` and `stage2_rollout_aligned` still use provenance-aware metric families, but the active single-pass Stage-2 contract now routes Channel-A through `loss/text/*`, `loss/coord/*`, and `coord_diag/*`, while Channel-B uses `loss/B_rollout_text/*`, `loss/B_coord/*`, and `coord_diag/B/*`.
  - Historical iterative groups such as `loss/A1_*`, `loss/A2_*`, `coord_diag/A1/*`, and `coord_diag/A2/*` are no longer part of the active Stage-2 contract.

## Stage-1 compact recursive detection and prefix roll-in

The active compact Stage-1 owner is the detection stack under `src/detection/`. There are now two distinct current-schema routes:

- `configs/stage1/recursive_detection_ce/prod/compact_full_support2.yaml` remains the random-permutation ET-RMP-CE production baseline/comparator.
- `configs/stage1/recursive_detection_ce/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml` is the active ablation successor config, with `cap8_frac0p06_mix0p1` as the slope ablation and `cap8_frac0p04_mix0p2` as the strength ablation; see [`INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`](INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md). These configs use type-gated schema/desc/coord/eos positions, structural boundary hard CE, description hard CE plus trie support/balance, and coord-vocab-only Gaussian SoftCE over active-branch remaining-instance mixtures. The coordinate target uses the focused R95 policy `floor(min(cap, fraction * axis_len))` with default cap8/4%/mix0.1, replacing the earlier unconstrained `sqrt(axis + 1)` wide-span draft behavior. Treat this successor as an ablation/smoke surface, not production-approved behavior, until production-scale validation evidence and explicit promotion approval are recorded.
- `configs/stage1/recursive_detection_ce/ablation/compact_full_prefix_rollin_balance2.yaml` is the E1 `prefix_rollin_et_rmp_ce` ablation route for Prefix-Closed Multi-Target SFT.
- `configs/stage1/recursive_detection_ce/ablation/compact_full_prefix_rollin_separator2.yaml` is the E2 separator-continue ablation. It keeps E1 support/balance/type-gate/EOS settings and changes only the append-boundary weights so the `\n` continuation token gets more pressure before `<|object_ref_start|>` can be emitted.

`prefix_rollin_et_rmp_ce` is compact-full only. It requires `detection_template.id: compact_full`, masks roll-in prefix labels, samples `K` uniformly over `[0, object_count]`, keeps `suffix_order: same_sampled_permutation` for V1, and expresses support/balance weights under `objective.target`, append-boundary weights under `objective.boundary`, and not obsolete flat trie-weight aliases.

EOS supervision for this variant targets the Qwen chat-template assistant stop marker `<|im_end|>` only. Text-level terminators such as `<|endoftext|>` or `<|end_of_text|>` must not be used as training EOS for this surface. The initial `empirical_unlabeled_poisson_v0` EOS prior is smoke/ablation-only: production configs must use `objective.eos.eos_trust_weight.source: calibrated_formula_ref` with a versioned calibration artifact and validation evidence.

Generation-time HF/Qwen surfaces use the global chat-token contract
`eos_token_id=id("<|im_end|>")` and `pad_token_id=id("<|endoftext|>")`.
This is a decode/runtime contract; it does not change the training target rule
above, where only `<|im_end|>` is the semantic EOS target. HF processor calls
for inference/rollout paths must preserve training-time geometry with
`do_resize=false`. vLLM local/server inference must also stop on
`"<|im_end|>"` only; do not add `<|endoftext|>` as a generation stop token.
Local vLLM launch kwargs should carry `mm_processor_kwargs: {do_resize: false}`
when the installed vLLM API supports it, and inference artifacts must record the
Qwen chat generation contract.

For `prefix_rollin_et_rmp_ce`, `objective.state_weighting` and
`objective.normalization` are authored config truth, not hidden runtime
substitutions. They must be `uniform_permutation` and
`semantic_image_bucket_balanced`, respectively. `training.effective_batch_size`
is likewise the source of truth for optimizer-step budget; YAML must not also
author `training.gradient_accumulation_steps`.

The append-boundary loss is also authored config truth. `objective.boundary`
must use `type: compact_full_append_boundary` and explicitly set
`separator_continue_weight`, `eos_stop_weight`, and `component_weight`. E1 keeps
the historical `0.5 / 0.5 / 0.3` boundary mix. E2 raises
`separator_continue_weight` to `2.0` while leaving `eos_stop_weight=0.5` and
`component_weight=0.3`, targeting the diagnosed failure where free decode stops
at `<|im_end|>` before emitting the required separator newline.

Compact-full token-row training uses 1002 trainable rows through the persisted
`coord_offset_adapter` module name: the 1000 coord rows plus
`<|object_ref_start|>` and `<|box_start|>`. Treat the persisted module name as
historical; the current contract is token-row adaptation, not coord-only
adaptation.

For `prefix_rollin_et_rmp_ce` smoke and ablation monitoring, use
`loss/recursive_detection_ce` as the comparable objective-loss scalar. The
top-level trainer `loss` may be scaled by gradient accumulation and is therefore
not directly comparable across different `training.effective_batch_size`
settings.

Required training-health diagnostics for this surface include:

- `recursive_detection_ce/target_mix/targets_per_sample`: how many supervised
  local next-token targets contributed to the logged optimizer step.
- `recursive_detection_ce/target_mix/eos_fraction` and
  `recursive_detection_ce/target_mix/non_eos_fraction`: whether a log row is
  dominated by easy `<|im_end|>` supervision or contains real continuation
  states.
- `recursive_detection_ce/target_mix/trie_multi_positive_fraction`: whether
  local multi-target object-entry supervision was actually present.
- `recursive_detection_ce/target_mix/coord_fraction`,
  `recursive_detection_ce/target_mix/desc_fraction`, and
  `recursive_detection_ce/target_mix/object_control_fraction`: schema/content
  composition for interpreting token accuracy and CE shifts.
- `recursive_detection_ce/target_mix/positive_children_per_trie_target`: the
  average branching factor for multi-positive entry targets.
- `recursive_detection_ce/trie_valid_mass`,
  `recursive_detection_ce/support_loss`, and
  `recursive_detection_ce/balance_loss`: support-vs-balance behavior for valid
  next-object entries. Healthy support should not collapse while balance remains
  nonzero enough to discourage one object from taking all probability mass.
- `recursive_detection_ce/entry/continue_minus_eos_margin`: the local
  continuation margin, computed as valid next-object log-mass minus the
  `<|im_end|>` logit. Positive values mean the model prefers continuing over
  stopping after a separator has already been supplied.
- `recursive_detection_ce/free_boundary/continue_minus_eos_margin` and
  `recursive_detection_ce/free_boundary/continue_mass`: the append-boundary
  signal for object separators, computed at the token where autoregressive
  decode must choose `\n` over `<|im_end|>` before the next object can start.
  This is the more direct early-stop health signal for non-empty prefixes.
- `recursive_detection_ce/boundary/separator_continue_weight`,
  `recursive_detection_ce/boundary/eos_stop_weight`, and
  `recursive_detection_ce/boundary/component_weight`: the runtime loss weights
  actually used by the trainer. These should match the materialized
  `objective.boundary` block so separator ablations are config-truthful.
- `recursive_detection_ce/entry/valid_child_entropy` and
  `recursive_detection_ce/entry/valid_child_kl_to_uniform`: whether valid
  children remain reasonably balanced or collapse to one easy object. The KL is
  `KL(Uniform(valid_children) || p_valid)`, so lower is more uniform.
- `detection_sequence/coordinate/token_acc/full_vocab/top1` and
  `detection_sequence/coordinate/token_ce/full_vocab`: coordinate-token
  learning pressure, which is often the hard part even when description/schema
  tokens look saturated.
- `recursive_detection_ce/eos_trust_weight`,
  `recursive_detection_ce/eos_unweighted_ce`, and
  `recursive_detection_ce/eos_weighted_loss`: whether the censored-EOS policy is
  weakening stop supervision as intended.
- `recursive_detection_ce/type_gate_loss`,
  `recursive_detection_ce/type_gate_allowed_mass`,
  `recursive_detection_ce/type_gate_allowed_tokens`, and
  `recursive_detection_ce/type_gate_weight`: schema/type safety pressure for
  keeping generated compact detections parseable.

Do not interpret a low aggregate loss or high `token_acc` as a healthy
multi-positive trend unless `target_mix/non_eos_fraction` and
`target_mix/trie_multi_positive_fraction` show that continuation and
multi-target positions were present in the logged rows. For tiny smoke runs,
prefer `effective_batch_size >= 4` when checking trend shape so uniformly sampled
`K in [0, N]` does not produce many EOS-only optimizer steps.

Use the forced-prefix continue-vs-EOS probe before changing EOS or balance
hyperparameters based on decode under-generation alone:

```bash
conda run -n ms python -m src.analysis.prefix_rollin_teacher_forced_diagnostic \
  --config configs/stage1/recursive_detection_ce/smoke/compact_full_prefix_rollin_tiny.yaml \
  --checkpoint outputs/stage1_2b/recursive_detection_ce/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664 \
  --split val \
  --limit 8 \
  --k-values every \
  --output-dir temp/prefix_rollin_forced_prefix_probe_limit8
```

The artifact is diagnostic-only and does not call `generate()`. Its
`per_case.jsonl` rows expose `prefix_k`, `prefix_mode`, `gt_count`,
`remaining_gt_count`, `continue_logsumexp`, `valid_mass`,
`continue_minus_eos_margin`, and `<|im_end|>` logits/log-probabilities.

Interpret the two continuation boundaries separately:

- `*_entry_after_separator` scores `<|object_ref_start|>` vs `<|im_end|>` after
  a separator newline has already been forced. This boundary can look extremely
  healthy while free decode still stops early.
- `*_free_boundary` scores the actual next token after the current object
  prefix: `\n` vs `<|im_end|>` for non-empty prefixes, or
  `<|object_ref_start|>` vs `<|im_end|>` for `K=0`. This is the boundary that
  explains early stopping in ordinary autoregressive decode.

To replay a free-decode artifact as the forced prefix, add the generated-prefix
mode:

```bash
conda run -n ms python -m src.analysis.prefix_rollin_teacher_forced_diagnostic \
  --config configs/stage1/recursive_detection_ce/smoke/compact_full_prefix_rollin_tiny.yaml \
  --checkpoint outputs/stage1_2b/recursive_detection_ce/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664 \
  --split val \
  --limit 8 \
  --prefix-modes generated_prefix \
  --decode-artifact temp/infer/recursive_detection_ce/smoke_compact_full_support2_tokenrows_v2_ckpt3664_hf_limit8/gt_vs_pred.jsonl \
  --trace-artifact temp/infer/recursive_detection_ce/smoke_compact_full_support2_tokenrows_v2_ckpt3664_hf_limit8/pred_token_trace.jsonl \
  --output-dir temp/prefix_rollin_generated_prefix_probe_limit8
```

If `entry_after_separator` is positive but `free_boundary` is negative, the
model knows how to start the next object after a newline but prefers
`<|im_end|>` over appending that newline. Treat this as a separator/append
continuation failure, not as evidence that the object-entry trie target itself
collapsed.

Current schema enforces production EOS source/reference shape. The stronger
content-level check that a calibration artifact is `production_approved` with a
full validation probe remains an artifact/registry gate until a concrete
validator is introduced.

Retired continuation code, config, and runtime paths should not be used for new training. Historical evidence remains in git history, archived progress notes, and run artifacts.

## Stage-1 non-canonical bbox V1 experiments

When `custom.bbox_format` is `cxcy_logw_logh` or `cxcywh`, the Stage-1 loss
surface is intentionally much narrower than the existing `xyxy` baseline:

- model-facing `bbox_2d` slots become either `[cx, cy, u(w), u(h)]` or
  `[cx, cy, w, h]`
- coord-token bbox supervision is hard CE only
- `custom.coord_soft_ce_w1.enabled` must remain `true`
- `ce_weight > 0`
- `soft_ce_weight = 0`
- `w1_weight = 0`
- `gate_weight > 0`
- `text_gate_weight > 0`
- `temperature = 1.0`, `target_sigma = 2.0`, and `target_truncate = null`
  are compatibility-only defaults and do not define a soft-label path here
- `custom.bbox_geo.*` and `custom.bbox_size_aux.*` are out of scope for this
  experiment

This V1 profile exists to isolate the parameterization question under minimal
loss complexity. It is not the same recipe as the legacy `xyxy` Stage-1
baseline documented above.

Evaluation note:

- standalone inference/eval still emits canonical pixel `xyxy` standardized
  artifacts
- official score-aware evaluation remains available through a deterministic
  constant-score `gt_vs_pred_scored.jsonl` compatibility artifact
- confidence post-op remains unsupported for these non-canonical formats in
  this V1 path
- checkpoints trained on the legacy model-facing `xyxy` serialization are not
  semantically compatible with non-canonical `infer.bbox_format`
- if you force an old `xyxy` checkpoint through the `cxcy_logw_logh` or
  `cxcywh` infer
  path, the runtime may still emit canonicalized `xyxy` artifacts, but those
  outputs should be treated as a compatibility stress test rather than as valid
  non-canonical rollout behavior
- training data for this profile must come from the offline-prepared
  derived preset root such as
  `public_data/<dataset>/<preset>_cxcy_logw_logh/train.coord.jsonl` or
  `public_data/<dataset>/<preset>_cxcywh/train.coord.jsonl` rather
  than a runtime reinterpretation of canonical preset JSONL

## Coord-offset adapter (tie-head / single shared table)

When training with coord tokens, CoordExp can optionally avoid updating the full vocabulary embedding
and instead learn a small **offset adapter** over just the coord-token id range.

**Key idea**:
- Freeze the base `embed_tokens.weight` and `lm_head.weight`.
- Train a compact offset table only for `<|coord_0|>.. <|coord_999|>` token ids.

**Config**:
```yaml
custom:
  coord_offset:
    enabled: true
    # Default: Qwen3-VL-style tie-head (single/shared lookup table for embed + head).
    tie_head: true
    ids: { start: 151670, end: 152669 }  # <|coord_0|>.. <|coord_999|>
    # Optional: learning-rate overrides for the offset parameters.
    # When tie_head: true, only embed_lr is used (head_lr is ignored).
    embed_lr: 1.0e-4
    head_lr: 1.0e-4
    weight_decay: 0.0
```

**Semantics**:
- `tie_head: true` (recommended; default)
  - The adapter trains a **single** offset table and uses it for both:
    - embedding lookup (adds offsets to hidden states for coord tokens), and
    - output projection (adds logits for coord tokens via `hidden @ offset^T`).
  - This is equivalent to applying a single delta to the tied embedding/head table for coord tokens,
    which matches the intended tie-head routine of Qwen-family LMs.
- `tie_head: false` (legacy/ablation)
  - Trains separate `embed_offset` and `head_offset` tables (two independent deltas).
  - Export/merge may need to materialize `lm_head.weight` and disable tying to preserve separate behavior.

**Export/merge**:
- Use `scripts/merge_coord.sh` to merge LoRA/DoRA and bake the coord-offset adapter into a merged HF checkpoint.
  - With `tie_head: true`, the merged checkpoint can keep tied embeddings (single table).
  - With `tie_head: false`, the merged checkpoint may need an explicit `lm_head.weight` tensor and `tie_word_embeddings: false`.
