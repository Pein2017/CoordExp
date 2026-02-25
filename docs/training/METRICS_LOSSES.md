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
      - `loss/A2_text/struct_ce`
      - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}`
      - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
    - Channel-B:
      - `loss/B_rollout_text/{struct_ce,desc_ce}`
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
    - `loss/A2_text/struct_ce`
    - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}`
    - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
  - Channel-B:
    - `loss/B_rollout_text/{struct_ce,desc_ce}`
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}`
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`

Reduction/naming contract:
- `loss/*` keys are mean-like scalars (comparable across packing/batch shapes).
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
- Canonical objective keys for this split include:
  - `loss/A1_text/{struct_ce,desc_ce}` (GT anchor forward; token CE objective atoms)
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

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) note (Channel-B path):
- Stage_2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) uses unified Channel-B by default:
  rollout prefix + FN injection + one teacher-forced forward with explicit CE masks.
- Channel-B CE/geometry semantics:
  - matched prefix: structure CE ON, desc/coord CE OFF,
  - FP prefix: structure/desc/coord CE OFF,
  - FN-injected: structure+desc CE ON, coord CE OFF,
  - geometry loss on matched + FN, FP geometry OFF.
- Legacy `reordered_gt_sft` has been removed (unified Channel-B only).

## Stage-2 Rollout-Matching Metrics (Training Logs)

Stage_2 (`custom.trainer_variant: stage2_rollout_aligned`) logs additional keys
under `rollout/*`, `packing/*`, and `time/*` to help diagnose failures and
performance during training.

For evaluation, Stage_2 uses a production-style evaluator (rollout -> parse ->
Hungarian match) and reports metrics under `eval_rollout/*` keys. This evaluator
intentionally skips teacher-forced encoding/loss computation, so `eval_loss` is
not reported for this trainer variant.

Important semantics:
- **Aggregated logging:** metrics are accumulated across gradient-accumulation micro-batches and
  logged once per optimizer step (same step index as `train/loss`).
- **Rank-local (training logs):** `rollout/*` keys logged during training are
  rank-local (not all-reduced), so they can vary across GPUs.
- **All-reduced (eval):** `eval_rollout/*` keys are aggregated over the full
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
  - **What:** effective sampling knobs used for rollout generation.
  - **Note:** vLLM backends currently enforce `decode_mode=greedy` as a **non-beam sentinel** (no beam support in this path),
    so `rollout/decode_non_beam_count == N` does **not** imply deterministic rollouts. Use `rollout/do_sample` and `rollout/temperature`
    to disambiguate deterministic vs sampling.

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
  - **What:** post-rollout packing stats (carry-only mode).

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
  - **What:** number of samples dropped because deterministic closure-marker resolution failed (`}` / `<|im_end|>` alignment).
  - **Why:** should remain ~0; non-zero indicates truncation or marker alignment issues.

- `stage2_ab/channel_b/invalid_rollout`
  - **What:** number of samples in this step where rollout parsing could not produce an append-ready `{"objects": [` prefix and therefore fell back to empty-pred mode.
  - **Why:** tracks sample-level rollout/container failures that force FN-only completion supervision.

Aggregation semantics (training-time `metrics` payload):
- counters are global sums across grad-accum + DDP ranks
- boolean activation flags use global max
- rates use ratio-of-global-sums (e.g., `rollout/parse_truncated_rate`)

## Stage-2 Rollout-Matching Metrics (Eval)

When `custom.trainer_variant: stage2_rollout_aligned` runs evaluation (`training.eval_strategy != no`),
it reports production-style metrics derived from rollout -> parse -> Hungarian matching.

Returned keys (prefixed with `eval_`):
- `eval_rollout/precision`, `eval_rollout/recall`, `eval_rollout/f1`
- `eval_rollout/pred_objects`, `eval_rollout/gt_objects`, `eval_rollout/matched`
- `eval_rollout/fp`, `eval_rollout/fn`
- `eval_rollout/parse_truncated_rate`
- `eval_rollout/parse_dropped_invalid`, `eval_rollout/parse_dropped_ambiguous`
- `eval_rollout/sample_valid_pred_rate`, `eval_rollout/sample_any_match_rate`
- `eval_rollout/matched_maskiou_mean`
- `eval_rollout/mAP` (when `rollout_matching.eval_detection.enabled: true`)
- `eval_rollout/coco_eval_ok` (1.0 on success, 0.0 on best-effort failure fallback)

COCO summary policy (eval-step):
- Only `eval_rollout/mAP` is emitted for COCO summary output.
- `eval_rollout/bbox_*` and `eval_rollout/segm_*` summary keys are intentionally suppressed during eval-step.

Optional desc monitor keys (when enabled):
- `eval_rollout/desc_pairs_total`
- `eval_rollout/desc_exact_acc_on_matched`
- `eval_rollout/desc_sem_enabled`
- `eval_rollout/desc_sem_acc_on_matched`
- `eval_rollout/desc_sem_sim_mean`, `eval_rollout/desc_sem_sim_count`

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

When `custom.coord_soft_ce_w1.enabled: true`:

1) Base LM loss (full vocab CE)
- **What:** model-native cross-entropy over the full vocabulary.
- **Where applied:** only on **non-coord** targets (coord targets are masked to `-100`).
- **Normalization:** mean over the number of supervised non-coord tokens (packing-safe).
- **In loss:** YES (train and eval_loss).

Diagnostics (logged as metrics, prefixed with `eval_` during evaluation):
- `base_ce/loss`
  - **What:** the base CE term after masking coord targets (i.e. non-coord CE only).
  - **In loss:** YES (this is the base term of the objective).
- `base_ce/noncoord_tokens`
  - **What:** supervised non-coord token count used by the base CE term (sanity-check denominator).
  - **In loss:** NO (diagnostic only).
- `base_ce/noncoord_tokens_per_sample`
  - **What:** `base_ce/noncoord_tokens / pack/num_samples` (packed runs only; batch-wide aggregate).
  - **In loss:** NO (diagnostic only; helps interpret scale per original sample).
- `base_ce/loss_per_sample`
  - **What:** approximate base-CE contribution per original sample:
    `base_ce/loss * base_ce/noncoord_tokens / pack/num_samples`.
  - **In loss:** NO (diagnostic only).

2) Coord-token loss (coord-gated distribution losses)
- **What:** extra supervision computed from the same forward logits at GT coord positions.
- **Where applied:** only at positions whose GT label is a coord token (1000-bin ordered vocab).
- **Normalization:** mean over the number of GT coord-token positions (packing-safe).
- **In loss:** YES (added to the base loss).

### Coord-token loss breakdown (`coord_diag/*`)

The coord loss is:

`coord_diag/loss = soft_ce_weight * softCE + w1_weight * W1 + ce_weight * CE + gate_weight * gate`

Compatibility note:
- The same breakdown is also logged under `coord_softce_w1/*` keys (legacy alias):
  - `coord_softce_w1/loss`, `coord_softce_w1/soft_ce`, `coord_softce_w1/w1`, `coord_softce_w1/ce`, `coord_softce_w1/gate`

- `coord_diag/enabled`
  - **What:** whether coord-gated softCE+W1(+gate) is active (`1.0`) or this is a pure-CE ablation (`0.0`).
  - **In loss:** NO (this is a tag for grouping/comparability across ablations).

- `coord_diag/loss`
  - **What:** coord loss term (already includes weights). When `coord_diag/enabled=1`, this is the
    coord term *added* to the training loss. When `coord_diag/enabled=0`, this is diagnostic-only.
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` (otherwise diagnostic-only).

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
  - **What:** number of micro-steps accumulated into the current optimizer step (≈ `gradient_accumulation_steps`).

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
