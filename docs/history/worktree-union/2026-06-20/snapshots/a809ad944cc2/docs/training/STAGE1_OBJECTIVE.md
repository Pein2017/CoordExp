---
doc_id: docs.training.stage1-objective
layer: docs
doc_type: reference
status: canonical
domain: training
summary: Stage-1 objective surfaces and coord-token training behavior.
updated: 2026-05-25
---

# Coord Objective & Adapter

This document details the specialized training objectives and architectural adapters used for coordinate tokens in CoordExp.

Scope note:
- This page is primarily the Stage-1 / baseline coord-objective reference.
- The canonical Stage-1 packing contract now lives in [`../data/PACKING.md`](../data/PACKING.md):
  one hard `global_max_length` cap, offline static packing, full-length probing before plan build,
  and fail-fast when any atomic sample exceeds the cap.
- For Stage-2 pipeline-declared training, the canonical objective surface now lives under:
  - `stage2_rollout_correction.pipeline` for `pipeline.id: stage2_rollout_correction`
- In those Stage-2 paths, active objective ownership is residual rollout
  correction through the pipeline surface described in:
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
- Legacy `custom.coord_soft_ce_w1.*` authoring should not be used for pipeline-declared Stage-2 configs.
- For standard Stage-1 SFT, the active non-pipeline teacher-forcing surface is:
  - `custom.coord_soft_ce_w1.*`
- Raw-text norm1000 ablations remain legacy Stage-1 SFT surfaces, not latest
  compact detection overlays. Materialization verification should include:
  - `configs/stage1/profiles/2b/raw_text_xyxy_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml`
- The canonical compact Stage-1 research teacher-forcing public route is
  `pipeline.id: stage1_research_teacher_forcing`; active configs live under
  `configs/stage1/detection_teacher_forcing/`.
- Compact prefix roll-in multi-positive training remains only a legacy
  recursive-detection CE comparator/ablation surface, not the active compact
  teacher-forcing route. The checked-in route
  the archived prefix-rollin recursive-detection config
  should be treated as E1 ablation/smoke validation, not production.
- Geometry-aware coordinate SoftCE for legacy compact recursive detection is
  scoped to historical A5-iou-gibbs/A6-ciou-gibbs and focused cap8
  instance-trie provenance configs under
  the archived recursive-detection production config root. It uses
  `objective.coord_soft_ce` and does not route through legacy
  `custom.coord_soft_ce_w1.*`.
- Narrow V1 exception:
  - `custom.bbox_format: cxcy_logw_logh` or `custom.bbox_format: cxcywh`
    defines an experimental Stage-1-only profile
  - under that profile, the allowed Stage-1 surface narrows to
    `custom.coord_soft_ce_w1.*` with hard CE plus positive coord/text gating
  - bbox geometry auxiliaries, soft CE, W1, and trainer-side rollout/Stage-2
    surfaces are intentionally out of scope and should be treated as invalid
    for that experiment

## Current Stage-1 Direction

The current public compact Stage-1 research teacher-forcing route is
`pipeline.id: stage1_research_teacher_forcing`; active configs live under
`configs/stage1/detection_teacher_forcing/`. Pipeline registry ids select the
public training family, while implementation ids preserve concrete descriptor
handles under `src/training/pipelines/`.

The new unified training architecture defines two Stage-1 pipeline ids:

- `pipeline.id: stage1_research_teacher_forcing`
  - public pipeline ID for compact-full Stage-1 objective research
  - implementation descriptor handle: `stage1_compact_trie_ce`
  - uses semantic compact template IDs such as `compact` or
    `compact_object_box_closed`; legacy config-level `compact_full` is a
    compatibility alias for semantic `compact`, not a low-level template id
  - supervises token spans, object-entry trie targets, coordinate soft targets,
    and optional decoded-box regression through typed objective atoms
  - canonical public compact teacher-forcing route; active configs under
    `configs/stage1/detection_teacher_forcing/` use
    `objective.id: research_teacher_forcing`
- `pipeline.id: stage1_standard_sft`
  - implementation descriptor handle: `stage1_json_ce`
  - JSON chat CE baseline for regression and fallback comparison
  - keeps the baseline teacher-forced JSON pipeline available without making it
    the compact-full default

The public registry objective profile order is:

```text
standard_ce, research_teacher_forcing, residual_set_correction
```

Objective authoring is keyed, but the registry emits this deterministic public
order. Internal term ids such as `token_ce`, `trie_ce`, and `coord_soft_ce` may
appear under `objective.terms` or concrete objective
modules/metrics only. Disabled terms remain explicit inside that objective
family so ablations preserve their intended contract instead of silently
deleting siblings.

Cleanup boundary:

- Compact-full Stage-1 should not reintroduce duplicate-burst unlikelihood,
  adjacent repulsion, EOS-loosen/trust/weighted-loss variants, continuation
  forcing, separator forcing, or stop-signal gate/damping variants.
- The removed names may appear in historical progress notes, diagnostic probes,
  compatibility readers, or absence tests only.
- Continue-vs-EOS probes remain diagnostic-only and must not be converted into
  production training mechanisms.

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
problem. Treat the older CE-side references as diagnostic evidence, not current
architecture guidance, unless a token-compatible pure-CE checkpoint is
evaluated under the same onset-local protocol.

## Coord distribution loss (coord tokens)

CoordExp can supervise coordinate tokens with **distribution-based losses** on
the legacy `xyxy` Stage-1 JSON baseline:

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
    #             + gate_weight * gate + text_gate_weight * text_gate
    ce_weight: 0.0
    soft_ce_weight: 1.0
    w1_weight: 1.0
    gate_weight: 1.0
    text_gate_weight: 0.0
    temperature: 1.0
    target_sigma: 2.0
    target_truncate: 16
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
  - Stage-1 coord-family loss keys include `coord_softce_w1/loss`, `coord_softce_w1/soft_ce`, `coord_softce_w1/w1`, `coord_softce_w1/gate`, and `coord_softce_w1/text_gate`
  - The older `coord_diag/*` diagnostic namespace is historical and is not an
    active coord-aux objective contract.
- Stage-2 note:
  - `stage2_rollout_correction` uses rollout-prefix roll-in plus residual/GT correction target IR metrics under `stage2_rollout_correction/*`.
  - Historical coord/bbox groups such as `loss/coord/*`, `loss/B_coord/*`, `loss/A1_*`, `loss/A2_*`, `coord_diag/*`, `coord_diag/A1/*`, and `coord_diag/A2/*` are no longer part of the active Stage-2 objective contract.

## Stage-1 compact detection teacher forcing and legacy recursive detection

The active compact Stage-1 owner is the detection stack under `src/detection/`.
The current public compact teacher-forcing route is
`pipeline.id: stage1_research_teacher_forcing`; active configs live under
`configs/stage1/detection_teacher_forcing/`.

The old recursive-detection CE configs below remain legacy/comparator,
migration, or ablation history only:

For ordinary baseline SFT, keep the standard `TrainingConfig` surface under
`configs/stage1/profiles/` rather than the detection teacher-forcing route.
Closed compact assistant rows are selected with
`custom.detection_template_id: compact_object_box_closed` while retaining
`custom.object_ordering: sorted` and Stage-1 static packing. For bbox-first
versus desc-first ablations, keep the same semantic template id and switch
`custom.object_field_order` / `detection_template.object_field_order` between
`geometry_first` and `desc_first`; cache fingerprints and infer artifacts record
both axes.

- `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/compact_full_support2.yaml` remains the random-permutation ET-RMP-CE legacy comparator, not the active compact teacher-forcing route.
- `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/compact_full_support2_iou_gibbs_softce_a5.yaml` is historical A5-iou-gibbs negative-result/superseded provenance: A2/support2 plus `iou_gibbs_v0` coordinate soft targets with `tau=0.0090909091` from the train one-token IoU-loss median.
- `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml` is historical paired A6-ciou-gibbs negative-result/superseded provenance: same setup as historical A5 but with `ciou_gibbs_v0`; production preparation assumed a separate 4-GPU slice for A5 and A6 rather than one 8-GPU run.
- `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml` is historical instance-trie/soft-CE ablation provenance, with `cap8_frac0p06_mix0p1` as the slope ablation and `cap8_frac0p04_mix0p2` as the strength ablation; see [`drafts/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`](drafts/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md). These configs are not the new typed teacher-forcing objective surface.
- the archived prefix-rollin recursive-detection config is the legacy/comparator E1 `prefix_rollin_et_rmp_ce` ablation route for Prefix-Closed Multi-Target SFT.

The legacy/comparator `prefix_rollin_et_rmp_ce` route is compact-row only. It
uses semantic `detection_template.id: compact` in current configs; old
`compact_full` config mentions are migration aliases or archive provenance. The
route masks roll-in prefix labels, samples `K` uniformly over `[0, object_count]`,
keeps `suffix_order: same_sampled_permutation` for V1, and expresses
support/balance weights under `objective.target`, not obsolete flat trie-weight
aliases.

EOS supervision for this variant targets the Qwen chat-template assistant stop marker `<|im_end|>` only, using ordinary teacher-forced CE. Text-level terminators such as `<|endoftext|>` or `<|end_of_text|>` must not be used as training EOS for this surface.

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

For legacy/comparator `prefix_rollin_et_rmp_ce`, `objective.state_weighting` and
`objective.normalization` are authored config truth, not hidden runtime
substitutions. They must be `uniform_permutation` and
`semantic_image_bucket_balanced`, respectively. `training.effective_batch_size`
is likewise the source of truth for optimizer-step budget; YAML must not also
author `training.gradient_accumulation_steps`.

Separator, terminal, and chat-stop positions are ordinary recursive CE targets.
They do not have a separate authored weighting section or metric; `<|im_end|>`
is supervised with the same teacher-forced CE path as other hard targets.

Compact token-row training uses the persisted `token_embeddings_adapter` module
name. The compact teacher-forcing route trains the 1000 coord rows plus required
schema rows such as `<|object_ref_start|>` and `<|box_start|>`; the
`compact_object_box_closed` route trains 1004 rows by also including
`<|object_ref_end|>` and `<|box_end|>`.

For legacy/comparator `prefix_rollin_et_rmp_ce` smoke and ablation monitoring, use
`loss/recursive_detection_ce` as the comparable objective-loss scalar. The
top-level trainer `loss` may be scaled by gradient accumulation and is therefore
not directly comparable across different `training.effective_batch_size`
settings.

Legacy recursive-detection CE training-health diagnostics for this comparator
surface include:

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
- `recursive_detection_ce/entry/valid_child_entropy` and
  `recursive_detection_ce/entry/valid_child_kl_to_uniform`: whether valid
  children remain reasonably balanced or collapse to one easy object. The KL is
  `KL(Uniform(valid_children) || p_valid)`, so lower is more uniform.
- `detection_sequence/coordinate/token_acc/full_vocab/top1` and
  `detection_sequence/coordinate/token_ce/full_vocab`: coordinate-token
  learning pressure, which is often the hard part even when description/schema
  tokens look saturated.
- `recursive_detection_ce/eos_unweighted_ce`: ordinary teacher-forced
  `<|im_end|>` CE at stop targets.
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

For analysis only, use the forced-prefix continue-vs-EOS probe before changing
EOS or balance hyperparameters based on decode under-generation alone. This is
a diagnostic probe, not a forced-continuation training mechanism:

```bash
conda run -n ms python -m src.analysis.prefix_rollin_teacher_forced_diagnostic \
  --config <archived recursive-detection smoke config> \
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
  --config <archived recursive-detection smoke config> \
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

The current training contract does not add an EOS calibration source or
production approval gate. Continue-vs-EOS probes are diagnostic only; training
keeps `<|im_end|>` as an ordinary teacher-forced CE target with weight `1.0`.

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
  are fixed defaults and do not define a soft-label path here
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

## Stage-1 raw-text xyxy benchmark

The minimal raw-text benchmark keeps canonical `xyxy` geometry and the shared
norm1000 lattice, but removes coord-token rendering:

Materialized legacy profile:

```text
configs/stage1/profiles/2b/raw_text_xyxy_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml
```

- train from canonical `train.norm.jsonl` / `val.norm.jsonl`
- set `custom.coord_tokens.enabled: false`
- keep `custom.coord_tokens.skip_bbox_norm: true`
- keep `custom.bbox_format: xyxy`
- keep `custom.coord_soft_ce_w1.enabled: false` for the pure-CE slice

Inference/eval for this benchmark must stay explicit:

- `infer.mode: text`
- `infer.pred_coord_mode: norm1000`
- `infer.bbox_format: xyxy`

Evaluation and visualization always canonicalize through
`norm1000 -> pixel-space xyxy` using the per-record image `width` and `height`
before drawing boxes or scoring metrics. Score-aware mAP for this benchmark
comes from numeric-span confidence post-op on the raw bbox integers rather than
from constant-score compatibility artifacts.

## Token-embeddings adapter (tie-head / single shared table)

When training with coord tokens, CoordExp can optionally avoid updating the full vocabulary embedding
and instead learn a small **token_embeddings_adapter** over role-resolved token rows.

**Key idea**:
- Freeze the base `embed_tokens.weight` and `lm_head.weight`.
- Train compact token-row offsets for `<|coord_0|>.. <|coord_999|>` and any
  schema tokens required by the active compact template.

**Config**:
```yaml
custom:
  token_embeddings_adapter:
    enabled: true
    # Default: Qwen3-VL-style tie-head (single/shared lookup table for embed + head).
    tie_head: true
    groups:
      coord_geometry:
        role: coord_geometry
        start_token: "<|coord_0|>"
        end_token: "<|coord_999|>"
        expected_start: 151670
        expected_end: 152669
      schema_tokens:
        role: structural_ce_only
        tokens:
          - "<|object_ref_start|>"
          - "<|object_ref_end|>"
          - "<|box_start|>"
          - "<|box_end|>"
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
- Use `scripts/merge_coord.sh` to merge LoRA/DoRA and bake the token-embeddings adapter into a merged HF checkpoint.
  - With `tie_head: true`, the merged checkpoint can keep tied embeddings (single table).
  - With `tie_head: false`, the merged checkpoint may need an explicit `lm_head.weight` tensor and `tie_word_embeddings: false`.
