---
doc_id: docs.training.metrics
layer: docs
doc_type: reference
status: canonical
domain: training
summary: Metric and loss-key interpretation for current training surfaces.
updated: 2026-03-09
---

# Training Metrics and Losses (Quick Reference)

This doc is a compact "API surface" for interpreting `Trainer.log()` keys produced during
training/evaluation runs.

Notes:
- Keys shown here are the *train* names. During evaluation, ms-swift prefixes keys with
  `eval_` (e.g. `coord_diag/loss` -> `eval_coord_diag/loss`).
- Unless otherwise stated, metrics are computed on **supervised next-token positions**
  (i.e. `labels[:, 1:] != -100`).
- Trainer metric updates flow through a neutral payload contract with explicit versioning:
  `schema_version` (integer major, current `1`), `mode` (`train|eval`), `global_step`,
  and `metrics` (key/value map). Missing/non-integer/unsupported schema versions are rejected.

## Namespace Hierarchy

Read metric keys left-to-right:

- `loss/<provenance>/<atom>`
  - Objective atoms that directly participate in the training loss.
  - `provenance` identifies the forward surface or supervision family, for example:
    - `A1_text`
    - `A1_coord`
    - `A2_text`
    - `A2_coord`
    - `B_rollout_text`
    - `B_coord`
  - `atom` is the post-weighting contribution name, for example `struct_ce`, `bbox_smoothl1`, or `coord_soft_ce`.

- `coord_diag/<metric>` and `coord_diag/<provenance>/<metric>`
  - Stage-1 uses bare `coord_diag/<metric>`.
  - Stage-2 two-channel uses provenance-split `coord_diag/A1/*`, `coord_diag/A2/*`, and `coord_diag/B/*`.

- `gradmon/<metric>` and `gradmon/<group>/<term>`
  - Optional loss-gradient monitoring diagnostics (sparse-emitted on monitor steps only).
  - `gradmon/loss_raw/*`, `gradmon/loss_ema_norm/*`, `gradmon/grad_norm/*`, and
    `gradmon/cos_to_total/*` are per-term gauges.
  - Aggregate keys such as `gradmon/neg_cosine_pair_frac` and
    `gradmon/grad_norm_ratio_max_over_median` summarize gradient domination/conflict.

- Training-time `rollout/<metric>` and eval families `eval/detection/*`, `eval/parsing/*`, `eval/description/*`, `eval/config/*`, `eval/runtime/*`
  - Training-time rollout telemetry uses `rollout/*`.
  - Eval-step rollout telemetry uses `eval/detection/*, eval/parsing/*, eval/description/*, eval/config/*, eval/runtime/*`.

- `packing/<metric>`, `time/<metric>`, `train/<metric>`, `accum/<metric>`
  - Operational telemetry families for packing, timing, sample accounting, and grad-accum/runtime state.

- `stage2/<metric>`, `stage2_ab/<...>`, `dup/<metric>`
  - Stage-2 scheduler / rollout-health tags, Channel-B-specific counters, and duplicate-collapse diagnostics.

- Channel-specific snapshots reuse the normal train-key hierarchy.
  - Stage-2 keeps the most recently observed Channel-A / Channel-B values and mirrors them into later
    `logging_step` rows, even when the current interval only contained the other channel.
  - There is no separate `latest/` namespace anymore; the stale-channel snapshot is written back under
    the same grouped key, for example `loss/A1_text/struct_ce` or `rollout/f1`.
  - Interpret a logged key as “most recent observed value for that metric at this logging step,” not
    strictly “computed in the immediately preceding interval.”

Naming rules:
- `/` separates namespace levels.
- `_` inside a leaf name is part of the metric name, not a hierarchy split.
- Suffixes like `_total`, `_count`, `_sum`, `_num`, `_den`, `_rate`, and `_frac` signal aggregation intent.
- Internal reducer helpers are underscore-prefixed path segments or leaf names such as `rollout/_parse_truncated_num`; they are not part of the final logged payload.

## Breaking change note (Stage-2 loss keys)

Stage-2 trainers (`custom.trainer_variant: stage2_two_channel|stage2_rollout_aligned`) now emit only
**objective atoms** under provenance keys (objective-by-default; no `*_obj` suffix or `obj/` prefix).

Definitions:
- An "objective atom" is a post-weighting contribution that directly participates in the trainer's total loss.
- Multi-term objective modules (notably bbox geometry and coord regularization) are **emitted as atoms**
  rather than as pre-summed aggregates (i.e., no `geo` / `coord_reg` combined keys).

Key replacements (non-exhaustive):
- Old objective suffix keys: `loss/token_ce_obj`, `loss/bbox_geo_obj`, `loss/coord_reg_obj` (removed)
  - Use provenance keys instead:
    - Channel-A:
      - `loss/A1_text/{struct_ce,desc_ce}`
      - Optional A1 anchor coord/geo atoms (only when configured via `bbox_geo.config.a1_*` / `coord_reg.config.a1_*`):
        - `loss/A1_coord/{bbox_smoothl1,bbox_ciou,coord_soft_ce,coord_w1}`
      - `loss/A2_text/struct_ce`
      - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}`
      - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
    - Channel-B:
      - `train/optimization/{loss_structure_ce,loss_description_ce,loss_dead_anchor_suppression}`
      - `loss/B_coord/{bbox_smoothl1,bbox_ciou}`
      - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
- Old provenance prefix: `obj/<provenance>/<atom>` (removed) -> `loss/<provenance>/<atom>`
- Legacy aliases removed: `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail`
- Rollout-only monitors are sparse-emitted: `rollout/*` and `time/rollout_*` keys are omitted on steps where no rollout ran.

## Stage-2 pipeline identity + metrics reduction contract

Pipeline identity (reproducibility):
- Stage-2 trainers resolve an effective objective/diagnostics pipeline at init and emit a stable
  `pipeline_checksum` together with resolved module lists in trainer init logs.
- `pipeline_checksum` is computed from canonical pipeline identity payload
  (`objective`, `diagnostics`, semantics-only `extra`) and is invariant to run context.
- The canonical `extra` keys used for checksum (when applicable) include:
  - `variant` (trainer variant name),
  - `stage2_ab.coord_ctx_embed_mode`, `stage2_ab.coord_decode_mode`,
  - `rollout_matching.coord_decode_mode`.
- Run context (`config`, `run_name`, `seed`) is logged separately and does not affect checksum.
- Treat the resolved module list + checksum as part of experiment identity for ablations.

ST bridge diagnostics surface:
- Stage-2 two-channel supports:
  - `stage2_ab.coord_ctx_embed_mode: soft|st|hard`
  - `stage2_ab.coord_decode_mode: exp|st`
- Stage-2 rollout-aligned supports:
  - `rollout_matching.coord_decode_mode: exp|st`
- For `L_geo` runs, use `stage2_ab.coord_ctx_embed_mode: st` for coord-slot embedding and
  `stage2_ab.coord_decode_mode: exp` (soft expectation decode) for geometry output decode.
- Canonical Stage-2 objective keys (objective atoms; emitted only when effective weight is non-zero) include:
  - Channel-A:
    - `loss/A1_text/{struct_ce,desc_ce}`
    - Optional: `loss/A1_coord/{bbox_smoothl1,bbox_ciou,coord_soft_ce,coord_w1}` (A1 anchor ablation knobs in pipeline module configs)
    - `loss/A2_text/struct_ce`
    - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}`
    - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
  - Channel-B:
    - `train/optimization/{loss_structure_ce,loss_description_ce,loss_dead_anchor_suppression}`
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}`
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`

Reduction/naming contract:
- `loss/*` keys are mean-like scalars (comparable across packing/batch shapes).
- `dup/max_desc_count` and `dup/saturation_rate` are mean-like duplicate-collapse gauges.
- Counter-like totals must use explicit suffixes (`*_total`, `*_count`, `*_sum`, `*_num`, `*_den`).
- Ratio-like keys should use `*_rate` / `*_frac`.
- Internal reduction helpers are underscore-prefixed (for example `rollout/_...`) and are removed
  from final logged payloads.

Stage-2 note (Stage-2 Rollout-Aligned Teacher Forcing):
- Stage_2 (`custom.trainer_variant: stage2_rollout_aligned`) uses masked losses on a single
  teacher-forced forward:
  - rollout prefix: coord-token supervision only (prefix text CE is masked out),
  - appended GT tail: normal CE on JSON structure, coord-token supervision on coord slots,
    and CE for `desc` string values is supervised by default (weighted via token_ce config,
    e.g. `desc_ce_weight` / `rollout_fn_desc_weight`).
- As a result, token-type metrics like `desc_token_frac` / `desc_token_acc` are meaningful
  for stage_2 runs when FN-appended tail desc tokens are present.
- Stage_2 runbook: `STAGE2_RUNBOOK.md`.

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) note (Channel-A path):
- Stage_2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) runs a
  two-surface objective in Channel-A:
  - **Anchor (GT / A1 logits):** full CE on JSON structure + `desc` value tokens (coord tokens excluded from CE).
  - **Self-context (final-iteration logits):** a format/closure CE stabilizer on `struct` + `<|im_end|>` only (no `desc` CE; coord tokens excluded from token-CE),
    with a small stabilizer weight controlled by `token_ce.config.self_context_struct_ce_weight` (pipeline-only; flat `stage2_ab.fmt_struct_ce_weight` removed).
  - Optional A1 coord/geo anchors (ablation knobs): enable via `bbox_geo.config.a1_*` / `coord_reg.config.a1_*` to add anchor supervision and emit `loss/A1_coord/*`.
- Canonical objective keys for this split include:
  - `loss/A1_text/{struct_ce,desc_ce}` (GT anchor forward; token CE objective atoms)
  - Optional: `loss/A1_coord/{bbox_smoothl1,bbox_ciou,coord_soft_ce,coord_w1}` (A1 anchor coord/geo atoms; see Stage-2 runbook)
  - `loss/A2_text/struct_ce` (final self-context forward; optional struct/EOS CE stabilizer)
  - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}` (final self-context forward; geometry objective atoms)
  - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (final self-context forward; coord_reg objective atoms)
- Channel-A coord losses are computed from the self-context logits (final iteration). By default, a small coord
  distribution regularizer may be enabled only for Channel-A via `coord_reg.config.self_context_soft_ce_weight`.
  Optional hard coord-token CE (peaky logits stabilizer) is controlled by `coord_reg.config.coord_ce_weight`.

Stage-2 coord-distribution diagnostics (`coord_diag/<provenance>/*`):
- These are **monitor-only** metrics derived from the coord-vocab slice at GT coord-token positions (not part of the loss).
- They are emitted only when `coord_diag` diagnostics module has a non-zero effective weight.
- Provenance:
  - `coord_diag/A1/*`: computed from Channel-A **A1** logits (GT anchor forward; `logits_a1`, `it==0`).
  - `coord_diag/A2/*`: computed from Channel-A **A2** logits (final softctx forward; `it==n_softctx_iter-1`), emitted only when `n_softctx_iter > 1`.
  - `coord_diag/B/*`: computed from Channel-B logits (rollout-context forward).
- Canonical keys under each provenance include:
  - `coord_diag/<prov>/coord_tokens_total`
  - `coord_diag/<prov>/{acc_top5,p_gt_mean,margin_mean,expected_bin_mae,expected_bin_abs_err_p90,coord_vocab_mass_mean}`

## Loss-Gradient Monitor (`gradmon/*`)

Enablement:
- Disabled by default.
- Opt in via `custom.extra.loss_gradient_monitor`.
- Sparse emission: keys appear only on monitor steps (default every 50 optimizer steps).

Canonical keys:
- Per-term gauges:
  - `gradmon/loss_raw/<term>`
  - `gradmon/loss_ema_norm/<term>`
  - `gradmon/grad_norm/<term>`
  - `gradmon/cos_to_total/<term>`
- Aggregate diagnostics:
  - `gradmon/grad_norm_ratio_max_over_median`
  - `gradmon/neg_cosine_pair_frac`
  - `gradmon/neg_cosine_pair_pct`
  - `gradmon/neg_cos_to_total_frac`
  - `gradmon/num_terms`
  - `gradmon/shared_param_count`
  - `gradmon/shared_param_numel`
- Timing:
  - `time/gradmon_s`

Term naming:
- Stage-1 coord-only monitor terms use:
  - `S1/coord_soft_ce`
  - `S1/coord_w1`
  - optional `S1/coord_ce`
  - optional `S1/coord_gate`
- Stage-2 rollout-aligned uses coord/geo atoms under `B_coord/*`.
- Stage-2 two-channel uses provenance-split coord/geo atoms:
  - `A1_coord/*`
  - `A2_coord/*`
  - `B_coord/*`
- Text CE and `text_gate` are excluded from `gradmon/*`.

Reduction contract:
- `gradmon/*` keys are best-effort diagnostics and do not affect optimization.
- Stage-2 trainers compute monitor metrics locally first, buffer them with the existing pending-log structures, and synchronize them only at the optimizer-step log boundary.
- `gradmon/*` gauges use the same observation-weighted reducer family as other mean-like diagnostics, without diluting sparse monitor steps across unobserved micro-packs.
- `time/gradmon_s` follows the active trainer's existing `time/*` reduction semantics.

Example config:

```yaml
custom:
  extra:
    loss_gradient_monitor:
      enabled: true
      interval_steps: 50
      ema_beta: 0.98
      require_sync_gradients: true
      coord_only: true
      granularity: atomic
      param_block:
        strategy: auto_last_lm_layernorm
        max_params: 64
        max_numel: 200000
```

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) note (Channel-B path):
- Stage_2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) now uses canonical clean-prefix Channel-B:
  anchor rollout + explorer rollout -> per-run strict accepted bbox objects -> per-run sequential dedup -> anchor/explorer triage -> anchor-edited clean prefix + weighted FN injection + dead-anchor duplicate UL.
- Channel-B CE/geometry semantics:
  - matched clean prefix: structure CE ON, desc/coord CE OFF,
  - shielded anchor objects: structure/desc/coord CE OFF,
  - FN-injected: structure+desc CE ON, coord CE OFF, with recovered GTs receiving higher per-object desc+geo+coord weight,
  - geometry loss on matched clean prefix objects + FN, shielded anchor objects OFF.
- Duplicate handling:
  - `train/optimization/loss_dead_anchor_suppression` is the explicit Channel-B dead-anchor suppression atom,
  - `loss_dead_anchor_suppression.config` is `{}` in v1 and the module `weight` is the only scaling surface,
  - dead anchor continuations are removed from the positive clean prefix and folded into boundary-local UL targets,
  - same-boundary dead continuations that share the same divergence token collapse to one UL term.
- Legacy `reordered_gt_sft` and `rollout_drop_invalid_struct_ce_multiplier` have been removed.

## Stage-2 Rollout-Matching Metrics (Training Logs)

Stage_2 (`custom.trainer_variant: stage2_rollout_aligned`) logs additional keys
under `rollout/*`, `packing/*`, and `time/*` to help diagnose failures and
performance during training.

For evaluation, Stage_2 uses a production-style evaluator (rollout -> parse ->
Hungarian match) and reports metrics under `eval/detection/*, eval/parsing/*, eval/description/*, eval/config/*, eval/runtime/*` keys. This evaluator
intentionally skips teacher-forced encoding/loss computation, so `eval_loss` is
not reported for this trainer variant.

Important semantics:
- **Aggregated logging:** metrics are accumulated across gradient-accumulation micro-batches and
  logged once per optimizer step (same step index as `train/loss`).
- **Buffered local -> globally reduced (training logs):** each rank computes and buffers training metrics locally during the step, then the trainer synchronizes the finalized step payload across DDP ranks at log time.
  - Counter-like keys reduce by global sum.
  - Mean-like `loss/*` / rollout gauges use the trainer's existing weighted-mean policy.
  - `time/*` tags follow the reducer family used by the active trainer (for example max-style reduction where the reducer already does that).
- **All-reduced (eval):** `eval/detection/*, eval/parsing/*, eval/description/*, eval/config/*, eval/runtime/*` keys are aggregated over the full
  evaluation dataset and summed across ranks.

### Rollout timing / throughput

- `time/rollout_generate_s`
- `time/rollout_parse_match_s`
- `time/rollout_teacher_encode_s`
  - **Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) note:** Channel-A steps do not generate rollouts, so these keys are omitted (absent; not `0.0`) on Channel-A steps.
    When diagnosing rollout throughput in AB-mixed runs, filter to steps where `stage2/channel_b == 1` or `stage2/raw_rollouts > 0`.

- `rollout/gen_new_tokens_total|mean|p90|p99`
  - **What:** generated assistant token counts (after stage_2 suffix trimming).
  - **Why:** helps detect "always hit max_new_tokens" pathologies.

- `rollout/gen_tokens_per_s`
  - **What:** `gen_new_tokens_total / time/rollout_generate_s`.
  - **Why:** detects rollout slowdowns (KV cache pressure / chunked prefill regressions).

### Decoding knobs (rollout generation)

Stage_2 uses `decode_mode` primarily to choose **beam** vs **non-beam** decoding.
Sampling is controlled separately via `decoding.temperature/top_p/top_k`.

- `rollout/do_sample`, `rollout/temperature`, `rollout/top_p`, `rollout/top_k`
  - **What:** backward-compatible anchor-side rollout tags for the active Channel-B step.
- `rollout/anchor_temperature`, `rollout/anchor_top_p`, `rollout/anchor_top_k`
- `rollout/explorer_temperature`, `rollout/explorer_top_p`, `rollout/explorer_top_k`
  - **What:** explicit dual-policy K=2 rollout tags for the anchor and explorer requests.
  - **Note:** vLLM backends support non-beam dual-policy rollouts in this path, but still reject `decode_mode=beam`.
    Use the anchor/explorer tags, not `rollout/decode_non_beam_count`, to disambiguate deterministic vs stochastic Channel-B behavior.

- `rollout/decode_non_beam_count`, `rollout/decode_beam_count`
  - **What:** counts of samples rolled out with `decode_mode != "beam"` (non-beam) vs `decode_mode == "beam"` (beam).
  - **Why:** helps detect accidental config drift across runs (e.g., beam search enabled unintentionally).

### Parse health

- `rollout/parse_dropped_invalid`, `rollout/parse_dropped_ambiguous`
  - **What:** number of predicted objects dropped by strict parsing.
- `rollout/parse_truncated`, `rollout/parse_truncated_rate`
  - **What:** sample count and rate where rollouts are truncated mid-object (suffix-trimmed).
- `rollout/parse_obj_total`
  - **What:** `valid_pred_objects + dropped_invalid + dropped_ambiguous` (object-level accounting).
- `rollout/parse_obj_valid_frac`, `rollout/parse_obj_drop_frac`
  - **What:** object-level valid/drop fractions.
- `rollout/sample_valid_pred_rate`
  - **What:** fraction of samples that yield at least one valid predicted object.
- `rollout/sample_any_match_rate`
  - **What:** fraction of samples that produce at least one supervised match.

### Matching quality (rollout-level)

- `rollout/gt_objects_total`, `rollout/valid_pred_objects_total`
  - **What:** GT and valid predicted object counts (post-parse; counter-like totals).
- `rollout/matched_for_supervision`, `rollout/excluded_from_supervision`
  - **What:** matched objects that were used vs excluded due to target-construction failure.
- `rollout/fn_appended_total`, `rollout/fn_total`
  - **What:** GT objects not supervised via prefix matching and aggregate FN totals (counter-like totals).
- `rollout/gating_rejections`, `rollout/gating_rejection_rate`
  - **What:** how often candidate pairs were rejected by the `maskiou_gate` threshold.

- `rollout/precision`, `rollout/recall`, `rollout/f1`
  - **What:** object-level precision/recall/F1 derived from matched-for-supervision.

- `rollout/matched_maskiou_mean`, `rollout/matched_maskiou_count`
  - **What:** mean maskIoU over matched pairs (norm1000-space, virtual canvas).
  - **Why:** disambiguates “more matches” vs “better geometry”.

### Desc monitoring (optional; metrics only)

Stage_2 can optionally monitor whether rollout `desc` strings stay semantically
aligned with GT on geometry-matched pairs. This is **metrics-only** and does not
affect the training loss.

- `rollout/desc_pairs_total`
  - **What:** number of geometry-matched pairs evaluated for desc monitoring.
- `rollout/desc_exact_acc_on_matched`
  - **What:** exact-match accuracy of normalized `desc` strings on matched pairs.
- `rollout/desc_sem_enabled`
  - **What:** 1.0 when the semantic embedding model is available for this step.
- `rollout/desc_sem_acc_on_matched`
  - **What:** semantic accuracy on matched pairs (exact OR cosine-sim >= threshold).
- `rollout/desc_sem_sim_mean`, `rollout/desc_sem_sim_count`
  - **What:** mean cosine similarity and count over matched pairs with embeddings.

### Supervision construction coverage

- `rollout/excluded_rate`
  - **What:** `excluded_from_supervision / (matched_for_supervision + excluded_from_supervision)`.
  - **Why:** detects OT/target-construction instability.

- `rollout/prefix_coord_targets_total`, `rollout/prefix_coord_targets_per_matched`
  - **What:** total coord slots supervised in the prefix and average per matched object.

- `rollout/tail_ignore_frac`
  - **What:** fraction of appended tail tokens that are ignored for CE due to `"desc"` masking.

### Length / packing diagnostics (stage_2)

- `rollout/prompt_len_mean|p90`
- `rollout/rollout_len_mean|p90`
- `rollout/train_len_mean|p90`
- `rollout/append_len_mean|p90`
- `rollout/encoded_len_mean|p90`
  - **What:** token-length summaries for prompt / rollout / training target / encoded length.
  - **Why:** explains OOM risk and packing fill changes between 1k/4k/8k max_new_tokens.

- `packing/post_rollout_fill`
- `packing/post_rollout_segments`
- `packing/post_rollout_buffer`
  - **What:** post-rollout packing stats for the teacher-forced forward pass.
  - **Note:** `stage2_rollout_aligned` uses a carry buffer across optimizer steps; `stage2_two_channel` is step-budgeted (no cross-step carry),
    so `packing/post_rollout_buffer` should normally drain to `0` at the end of each step.

### Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) schedule telemetry

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) logs a small set of scheduler diagnostics
once per optimizer step (aggregated across gradient accumulation):

- `stage2_ab/b_ratio_realized`
  - **What:** rolling realized fraction of optimizer steps that executed Channel-B.
  - **Why:** sanity-check that the deterministic Bresenham schedule matches `stage2_ab.schedule.b_ratio` over time.

### Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) Channel-B strict-drop and closure-supervision diagnostics

These keys are emitted by `custom.trainer_variant: stage2_two_channel` during Channel-B construction.

- `stage2_ab/channel_b/strict_drop/N_valid_pred`
  - **What:** number of predicted objects retained after strict validation for this step.
  - **Why:** numerator for strict-drop health checks.

- `stage2_ab/channel_b/strict_drop/N_drop_invalid`
  - **What:** number of predicted objects dropped by strict validation for this step.
  - **Why:** tracks parser/data-quality pressure on Channel-B supervision.

- `stage2_ab/channel_b/strict_drop/reason/<bucket>`
  - **What:** reason-bucket counts for dropped predictions (e.g., `missing_desc`, `wrong_arity`, `bbox_invalid`).
  - **Why:** identifies dominant failure modes in rollout outputs.

- `stage2_ab/channel_b/closure_supervision/N_drop`
  - **What:** legacy-named counter for samples that hit closure-resolution fallback because deterministic closure-marker bookkeeping failed (`}` / `<|im_end|>` alignment).
  - **Why:** should remain ~0; non-zero indicates truncation or marker-alignment issues, but the sample is still kept on the normal FN-tail supervision path.

- `stage2/invalid_rollout`, `stage2_ab/channel_b/invalid_rollout`
  - **What:** number of samples in this step where the rollout was marked invalid for Channel-B supervision construction.
    - Includes parser-level invalid rollouts that fall back to the canonical empty prefix.
    - Does **not** include closure-resolution fallback activations; those stay under `stage2_ab/channel_b/closure_supervision/N_drop`.
  - **Why:** tracks parser/container failures that force FN-only completion supervision on an empty clean prefix.

### Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) Channel-B duplicate-collapse diagnostics

- `dup/max_desc_count`
  - **What:** batch-mean of the per-sample maximum normalized-desc multiplicity in raw bbox-valid Channel-B predictions.
  - **Why:** a direct near-duplication gauge; should fall when clean-prefix training is working.

- `dup/saturation_rate`
  - **What:** batch-mean fraction of raw bbox-valid Channel-B predictions containing at least one coord at `0` or `999`.
  - **Why:** boundary-drift / saturation proxy that often rises with duplicate-heavy truncation.

- `dup/near_iou90_pairs_same_desc_count`
  - **What:** count of raw bbox-valid prediction pairs with identical normalized desc and `IoU >= 0.90`.
  - **Why:** targeted duplicate counter.

- `dup/near_iou90_pairs_any_desc_count`
  - **What:** count of raw bbox-valid prediction pairs with `IoU >= 0.90` regardless of desc.
  - **Why:** broader self-collision counter.

- `stage2_ab/channel_b/dup/N_raw_bbox_valid`
  - **What:** total number of raw bbox-valid parsed Channel-B predictions before sequential dedup.

- `stage2_ab/channel_b/dup/N_clean_accepted`
  - **What:** total number of deduplicated clean accepted Channel-B objects used for matching and positive teacher forcing.

- `stage2_ab/channel_b/dup/N_duplicates`
  - **What:** total number of duplicate objects removed from the positive clean prefix.

- `stage2_ab/channel_b/dup/N_duplicate_bursts`
  - **What:** total number of clean boundaries that received at least one duplicate burst.

- `stage2_ab/channel_b/dup/N_ul_boundaries`
  - **What:** total number of clean boundaries that contributed at least one retained duplicate-UL term.

- `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence`
  - **What:** total number of duplicate continuations skipped because no safe divergence token could be used.
  - **Why:** tokenizer/serialization safety check; should stay low.

### Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) Channel-B triage diagnostics

- `train/triage/gt_backed_count`
  - **What:** total number of anchor clean objects kept as GT-backed positives.

- `train/triage/unlabeled_consistent_count`
  - **What:** total number of anchor clean objects retained as neutral shielded context.

- `train/triage/dead_anchor_count`
  - **What:** total number of anchor clean objects removed from the positive prefix and sourced into dead-anchor suppression.

- `train/triage/explorer_only_dead_count`
  - **What:** total number of explorer-side non-GT-backed objects that remained unretained after triage.

- `train/triage/recovered_ground_truth_count`
  - **What:** total number of GT objects missed by anchor but recovered by the explorer rollout.

- `train/triage/recovered_ground_truth_rate_num`
- `train/triage/recovered_ground_truth_rate_den`
  - **What:** additive numerator/denominator pair for recovered-GT rate calculations.

- `train/triage/recovered_ground_truth_rate`
  - **What:** `recovered_ground_truth_count / recovered_ground_truth_rate_den`.
  - **Interpretation:** share of FN-tail GT objects that were recovered by the explorer rollout.

- `train/triage/dead_anchor_rate_num`
- `train/triage/dead_anchor_rate_den`
  - **What:** additive numerator/denominator pair for dead-anchor rate calculations.

- `train/triage/dead_anchor_rate`
  - **What:** `dead_anchor_count / dead_anchor_rate_den`.
  - **Interpretation:** share of anchor clean objects removed from the positive prefix.

- `train/triage/explorer_only_dead_rate_num`
- `train/triage/explorer_only_dead_rate_den`
- `train/triage/explorer_only_dead_rate`
  - **What:** explorer-only dead-object numerator/denominator pair plus the derived rate.
  - **Interpretation:** how much of the explorer clean set was mined but not kept as GT-backed or shielded context.

- `rollout/anchor/*` and `rollout/explorer/*`
  - **What:** policy-split rollout telemetry for the same Channel-B window.
  - **Includes:** `pred_objects`, `valid_pred_objects`, `parse_truncated_rate`, `gen_new_tokens_mean`, `gen_new_tokens_p90`, `near_iou90_any`, `near_iou90_same`.
  - **Explorer-only includes:** `temperature`, `do_sample`, `top_p`, `top_k`.
  - **Interpretation:** these are the diagnostics to compare conservative anchor behavior against stochastic explorer behavior without overloading the legacy flat `rollout/*` keys.

- `loss/B_rollout_text/dead_anchor_suppression`
  - **What:** compatibility alias for the Channel-B dead-anchor UL atom under the `loss/*` namespace.
  - **Why:** keeps dead-anchor suppression inspectable beside `struct_ce` / `desc_ce`.

- `diag/dead_anchor/num_terms`
- `diag/dead_anchor/num_ul_boundaries`
- `diag/dead_anchor/loss_per_term`
  - **What:** per-step dead-anchor UL diagnostics.
  - **Interpretation:** `num_terms` counts retained UL targets, `num_ul_boundaries` counts the contributing clean boundaries, and `loss_per_term` is the mean UL term value for the current step.

- `rollout/matched_for_supervision_count`
- `rollout/matched_for_supervision_over_valid_pred`
  - **What:** direct productive-supervision aliases for matched clean predictions.
  - **Interpretation:** use `matched_for_supervision_over_valid_pred` as a quick efficiency gauge for how much of the strict-valid predicted set contributes positive supervision.

- `stage2_ab/channel_b/train_monitor_dump_written`
  - **What:** `1` when the current logged Channel-B step successfully wrote a train monitor dump, else `0`.
  - **Why:** confirms that `monitor_dumps/` emission is actually happening at the configured cadence.

### Stage-2 Two-Channel Rollout Tags (Channel-B Steps Only)

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) emits
additional per-step tags on Channel-B steps to make rollout configuration and strict-drop pressure visible in logs.

- `stage2/raw_rollouts`
  - **What:** number of raw rollouts produced in this step (0 on Channel-A steps).
  - **Why:** disambiguates “no rollout happened” vs “rollout happened but produced no valid objects”.

- `stage2/invalid_rollout`
  - **What:** number of rollouts that were marked invalid (see strict-drop section above for details).
  - **Why:** quick health check for container/prefix/closure failures; should stay near 0.

- `stage2/drop_poly`
  - **What:** number of strictly-parsed predicted objects dropped because they were `poly` (Channel-B currently supports `bbox_2d` only).
  - **Why:** detects schema drift when rollouts emit polygons unexpectedly.

- `stage2/drop_unknown`
  - **What:** number of strictly-parsed predicted objects dropped due to unknown geometry types.
  - **Why:** catches unexpected geometry emission.

- `stage2/drop_bbox_invalid`
  - **What:** number of strictly-parsed predicted objects dropped due to invalid bbox arity/order (e.g. wrong length, non-int bins, or `x2<x1`/`y2<y1`).
  - **Why:** detects bbox decoding / truncation pathologies.

- `rollout/backend_hf`, `rollout/backend_vllm`
  - **What:** backend identity tags (1.0 on the active backend).

- `rollout/decode_mode_greedy`, `rollout/decode_mode_beam`
  - **What:** decode-mode tags (1.0 on the active mode).

- `rollout/seed_base`, `rollout/hf_seeded_global`
  - **What:** seed tags used for best-effort determinism diagnostics.

- `rollout/max_new_tokens`, `rollout/num_beams`, `rollout/repetition_penalty`
  - **What:** effective rollout decode knobs (logged as scalars for run provenance/debugging).

Aggregation semantics (training-time `metrics` payload):
- counters are global sums across grad-accum + DDP ranks
- boolean activation flags use global max
- rates use ratio-of-global-sums (e.g., `rollout/parse_truncated_rate`)
- for Stage-2 AB step-budgeted packing, mean-like scalars are segment-weighted across micro-packs (internal `stage2/_log_weight`).

## Stage-2 Rollout-Matching Metrics (Eval)

When `custom.trainer_variant: stage2_rollout_aligned` runs evaluation (`training.eval_strategy != no`),
it reports production-style metrics derived from rollout -> parse -> Hungarian matching.

Returned keys (prefixed with `eval_`):
- `eval/detection/precision`, `eval/detection/recall`, `eval/detection/f1`
- `eval/detection/pred_objects`, `eval/detection/gt_objects_total`, `eval/detection/matched`
- `eval/detection/fp_total`, `eval/detection/fn_total` (aliases: `eval/detection/fp`, `eval/detection/fn`)
- `eval/parsing/parse_truncated_rate`
- `eval/parsing/parse_dropped_invalid`, `eval/parsing/parse_dropped_ambiguous`
- `eval/parsing/sample_valid_pred_rate`, `eval/detection/sample_any_match_rate`
- `eval/detection/matched_maskiou_mean`
- `eval/detection/mAP` (when `rollout_matching.eval_detection.enabled: true`)
- `eval/runtime/coco_eval_ok` (1.0 on success, 0.0 on best-effort failure fallback)
- `eval/runtime/coco_counter_*` (compact COCO failure counters, when detection eval runs)
- `eval/config/prompt_variant_is_coco_80` (1.0 iff `rollout_matching.eval_prompt_variant: coco_80`)

COCO summary policy (eval-step):
- COCO eval logs `eval/detection/mAP` plus a small set of `eval/runtime/coco_counter_*` counters.
- Per-class summaries and `eval/detection/bbox_*` / `eval/detection/segm_*` aggregates are intentionally suppressed during eval-step.

Optional desc monitor keys (when enabled):
- `eval/description/desc_pairs_total`
- `eval/description/desc_exact_acc_on_matched`
- `eval/description/desc_sem_enabled`
- `eval/description/desc_sem_acc_on_matched`
- `eval/description/desc_sem_sim_mean`, `eval/description/desc_sem_sim_count`

Config tip:
- For Stage_2 runs, prefer `training.metric_for_best_model: rollout/f1` (and
  `training.greater_is_better: true`) to select best checkpoints by on-policy rollout quality.

## Loss Composition (Stage-1 / Scheme A)

Coord-offset adapter note (tie-head):
- Some stage-1 configs use `custom.coord_offset.enabled: true` to train coord-token rows via a lightweight
  offset adapter (instead of updating the full vocab embedding/head).
- By default `custom.coord_offset.tie_head: true`, which enforces a **single shared offset table** used for
  both embedding lookup and lm_head logits (Qwen-family tie-head semantics).
- Set `custom.coord_offset.tie_head: false` only for ablations that intentionally train embedding vs head
  offsets separately (export/merge then may need to materialize `lm_head.weight` and disable tying).

Legacy coord-loss note:
- `custom.coord_soft_ce_w1.*` is not part of the live public contract.
- Canonical coord supervision now flows through the pipeline modules, primarily `coord_reg`, with diagnostics such as `coord_diag`.
- Do not expect legacy `coord_softce_w1/*` aliases or implicit aux-loss defaults on the latest Stage-2 and rollout-aligned surfaces.

- `coord_diag/soft_ce`
  - **What:** soft cross-entropy between the predicted coord distribution and a Gaussian
    soft target centered at the GT bin.
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` (otherwise diagnostic-only).

- `coord_diag/w1`
  - **What:** 1D Wasserstein-1 distance computed via CDF differences on the ordered bins.
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` (otherwise diagnostic-only).

- `coord_diag/ce`
  - **What:** optional CE-on-bins term (pure cross-entropy over the 1000-bin coord vocab).
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` and `ce_weight != 0` (otherwise diagnostic-only).

- `coord_diag/gate`
  - **What:** coord-vocab "mass leak" penalty at GT coord positions:
      `gate = -log(sum_{i in coord_vocab} softmax(full_logits / T)[i])`
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` and `gate_weight != 0` (otherwise diagnostic-only).

- `coord_diag/coord_vocab_mass`
  - **What:** mean probability mass inside the coord sub-vocabulary at GT coord positions.
    This is derived from the gate computation (approximately `exp(-gate)`).
  - **In loss:** NO (diagnostic only).

- `coord_diag/coord_tokens`
  - **What:** number of GT coord-token positions in the current batch (or mean count over
    logging windows). Useful for sanity-checking the denominator.
  - **In loss:** NO (diagnostic only).

- `coord_diag/acc_top5`
  - **What:** top-5 accuracy within the 1000-bin coord sub-vocabulary at GT coord positions.
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/p_gt_mean`
  - **What:** mean predicted probability assigned to the GT coord bin (after temperature).
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/margin_mean`
  - **What:** mean `(max_logit - gt_logit)` within coord vocab (after temperature); lower is better.
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/expected_bin_mae`
  - **What:** mean absolute error between the expected coord bin index (under the predicted
    coord-vocab distribution) and the GT bin index. Units are **bins** (0..999).
  - **In loss:** NO (distribution-quality monitor; often more informative than top-k early on).

- `coord_diag/expected_bin_abs_err_p90`
  - **What:** 90th percentile (p90) of the per-token absolute error
    `abs(expected_bin - gt_bin)` at GT coord positions. Units are **bins** (0..999).
  - **Why:** tail-sensitive complement to `expected_bin_mae` (mean-only summaries can hide heavy tails).
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/w1_to_delta`
  - **What:** W1 distance from the predicted coord distribution `p(k)` to a delta at the GT bin `t`,
    i.e. `E_p[|k - t|]`. Units are **bins** (0..999).
  - **Why:** expectation-friendly proxy for continuous geometry: distinguishes "flat but not closer"
    from "flat and near-GT mass". More shape-sensitive than `abs(E[k] - t)`.
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/coord_tokens_per_sample`
  - **What:** `coord_diag/coord_tokens / pack/num_samples` (packed runs only; batch-wide aggregate).
  - **In loss:** NO (diagnostic only).

- `coord_diag/loss_per_sample`
  - **What:** approximate coord-loss contribution per original sample:
    `coord_diag/loss * coord_diag/coord_tokens / pack/num_samples`.
  - **In loss:** NO (diagnostic only).

- `stage1/total_loss_per_sample_est`
  - **What:** `base_ce/loss_per_sample + coord_diag/loss_per_sample` (approx. total per-sample objective).
  - **In loss:** NO (diagnostic only; useful for packed runs).

## Token Accuracy and Token-Type Metrics

These metrics help interpret "does it output the correct token id", separated by token
categories. They are all **metrics-only** (not part of the loss).

**Config**:
- Enable with `custom.token_type_metrics.enabled: true`.
- Defaults: `include: ["lvis"]`, `exclude: []`.
- Set `custom.token_type_metrics.log_top5: false` to skip top-k metrics (can reduce logging overhead).
- Works on padded and packed batches: token types are computed per sample pre-pack and concatenated; if alignment fails the metrics are skipped (training continues).
- NaN-safe: batches with zero supervised tokens are skipped.

- `token_acc` (from ms-swift)
  - **What:** top-1 token accuracy over supervised tokens (argmax vs GT).
  - **In loss:** NO.

- `token_acc_top5` (CoordExp aggregate metric)
  - **What:** top-5 token accuracy over supervised tokens.
  - **In loss:** NO.
  - **Config:** set `custom.token_type_metrics.log_top5: false` to skip top-k metrics (can reduce logging overhead).

- `text_token_acc`
  - **What:** top-1 accuracy over supervised tokens that are not GT coord tokens.
  - **In loss:** NO.

If `custom.token_type_metrics.enabled: true`, we also log per-type splits (all metrics-only):
- `desc_token_frac`, `format_token_frac`, `coord_token_frac`
  - **What:** fraction of supervised tokens belonging to each GT type.

- `desc_token_acc`, `format_token_acc`, `coord_token_acc`
  - **What:** top-1 accuracy within each GT type.

- `desc_token_acc_top5`, `format_token_acc_top5`, `coord_token_acc_top5`
  - **What:** top-5 accuracy within each GT type.

## Coord Vocab "Both Ways" Monitors (`coord_monitor/*`)

These are diagnostics to disambiguate two failure directions:
- GT coord slot -> predicted non-coord token (coord slot collapse)
- GT non-coord slot (format/desc) -> predicted coord token (coord intrusion)

All `coord_monitor/*` keys are **metrics-only** (not part of the loss).

### Type flip rates (argmax-based)

- `coord_monitor/flip_coord_to_noncoord`
  - **What:** among GT coord slots, fraction where argmax prediction is NOT a coord token.

- `coord_monitor/flip_text_to_coord`
  - **What:** among GT non-coord slots, fraction where argmax prediction IS a coord token.

- `coord_monitor/flip_format_to_coord`
  - **What:** among GT FORMAT slots (e.g. `}`, `]`, `:`), fraction predicted as coord.

- `coord_monitor/flip_desc_to_coord`
  - **What:** among GT DESC slots, fraction predicted as coord.

### Coord-vocab mass (softmax-mass-based)

These report mean probability mass inside the coord sub-vocabulary, conditioned on GT type:

- `coord_monitor/coord_vocab_mass_at_gt_coord`
- `coord_monitor/coord_vocab_mass_at_gt_text`
- `coord_monitor/coord_vocab_mass_at_gt_format`
- `coord_monitor/coord_vocab_mass_at_gt_desc`

Interpretation:
- Low `..._at_gt_coord` means coord slots are bleeding probability into non-coord tokens.
- High `..._at_gt_format` / `..._at_gt_desc` means coord tokens are intruding into text/format slots.

Config:
- Set `custom.token_type_metrics.coord_monitor_mass: false` to skip these mass diagnostics.
- Optionally cap compute cost with `custom.token_type_metrics.coord_monitor_mass_max_tokens: <int>` (0 = no cap).

## Packed-Run Per-Sample Helpers

When packing is enabled, one "training unit" is a concatenation of multiple original samples.
To make scales more intuitive, CoordExp logs a pack-size helper:
- `pack/num_samples`
  - **What:** number of original samples concatenated into the current unit (batch-wide aggregate).
  - **Note:** in non-packed runs, this is effectively the batch size.

Stage_2 (Stage-2 Rollout-Aligned Teacher Forcing / Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout)) also logs packing-aware step helpers that are stable even when
post-rollout packing is used:
- `train/samples_total`
  - **What:** total number of raw (unpacked) samples that contributed to the current optimizer step.
  - **Channel-A:** dataset samples packed into the learner sequences.
  - **Channel-B:** rollout samples packed into the learner sequences (after parse/match/FN-append).
- `train/samples_seen`
  - **What:** cumulative `train/samples_total` over the run (rank-local in multi-GPU; exact in server-mode world_size=1).
  - **Why:** use this as a packing-aware "progress" axis for eval scheduling and throughput comparisons.
- `train/micro_steps`
  - **What:** number of packed forward/backward passes executed inside the current optimizer step (i.e., number of packed sequences consumed).
  - **Why:** directly measures how many times the model did a full forward/backward for this step; post-rollout packing selection aims to
    minimize this under step-budgeted semantics.

## Reduction / Aggregation Semantics (Important)

Internally, per-step values are pushed into ms-swift's `MeanMetric` containers.
For scalar floats, this typically means "average over logging steps" (not token-weighted).

Where a metric is intended to be token-weighted, CoordExp computes a per-token mean first,
then updates the metric with a scalar.

Grad-accumulation note:
- Train `loss` is intended to be comparable to `eval_loss` (mean objective per optimizer step), even when
  `gradient_accumulation_steps > 1`.
- CoordExp logs two runtime helpers (metrics-only; prefixed with `eval_` during eval):
  - `accum/grad_steps`: configured `gradient_accumulation_steps`.
  - `accum/current_grad_steps`: per-update value (may differ on the last partial update in an epoch).
