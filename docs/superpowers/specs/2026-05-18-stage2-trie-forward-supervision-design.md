# Stage-2 Trie Forward Supervision Design

Date: 2026-05-18

Status: design approved for implementation planning. This is a Superpowers roadmap artifact, not a stable OpenSpec contract. Stable docs and OpenSpec updates should happen after the first tiny overfit evidence proves the behavior.

Parent roadmap:

- `docs/superpowers/specs/2026-05-15-unified-training-infrastructure-architecture-design.md`
- `docs/superpowers/plans/2026-05-15-unified-training-infrastructure-architecture-refactor.md`

Implementation plan:

- `docs/superpowers/plans/2026-05-18-stage2-trie-forward-supervision.md`

## Problem Statement

The current Stage-2 two-channel training loop can launch, roll out compact-full predictions, pair predictions with ground truth through greedy IoU, insert false negatives, and supervise a repaired Channel-B target. However, the current Channel-B forward objective still behaves like a single repaired sequence SFT objective. That makes the model pay for one sampled ordering and one repaired target path, even though Stage-2 data construction naturally produces multiple valid alternatives:

- K rollout-derived candidates per image.
- Different false-negative insertion policies.
- Matched predictions that can be preserved in context.
- Unmatched false positives that should be visible for diagnosis without always becoming hard negatives.
- Invalid or empty rollouts that should receive corrective fallback supervision without being counted as healthy rollout evidence.

This is misaligned with the current research direction. Stage-1 already treats entry-trie multiple-positive CE as a core training primitive. Stage-2 should reuse the same principle: build a per-example trie over valid repaired Channel-B continuations and train against the set of acceptable next tokens instead of a single serialized answer.

## Target Outcome

Stage-2 Channel-B gets a new objective module, `stage2_trie_ce`, that mirrors Stage-1 trie multiple-positive CE while preserving Stage-2's rollout-aware evidence path.

The conceptual flow is:

```text
image + prompt
  -> K unconstrained compact-full self-rollouts
  -> parse compact-full objects
  -> greedy IoU pairing against GT
  -> repair candidates with matched objects and false-negative insertions
  -> retain neutral or weak-positive false-positive context according to policy
  -> compile one per-example token trie over all valid candidates
  -> compute hard trie CE on Channel-B logits
  -> log span/object/rollout/fallback diagnostics
```

The goal is not to hide Stage-2 rollout failures. Invalid sequences, empty rollouts, fallback supervision, duplicate bursts, and weak false-positive evidence must remain visible in metrics and artifacts. The objective should make those cases trainable without making them invisible.

## Scope

In scope:

- Stage-2 two-channel compact-full Channel-B objective.
- Greedy-IoU paired rollout candidates.
- K=4 candidate aggregation by default.
- Pure hard CE Stage-2 trie loss with `token_mean` normalization.
- False-negative insertion policy ablations: `tail_append`, `fn_slot_shuffle`, `sorted`.
- False-positive policy ablations: `zero_loss_context`, `weak_positive_context`.
- Downweighted fallback corrective supervision.
- Span/token/object diagnostics for future thresholding and continuous objectives.
- Config-first selection through `stage2_ab.pipeline.objective`.

Out of scope for this implementation slice:

- Production-scale evaluation.
- Constrained decoding in the training path.
- Stable OpenSpec migration before evidence.
- New continuous coordinate or geometry losses.
- Unlabeled/pseudo-positive threshold filtering.
- Changes to upstream Hugging Face or Qwen3-VL model files.

## Confirmed Design Decisions

### Objective Shape

Stage-2 trie v0 is a pure hard CE objective. It owns Channel-B structural tokens, description tokens, coordinate tokens, object-entry alternatives, continuation alternatives, and EOS alternatives. It does not use bbox-GIoU/CIoU, bbox size auxiliary loss, coordinate regression, soft CE, Wasserstein coordinate loss, or object-level continuous gates.

The module must keep an explicit interface boundary so future continuous modules can consume the same Stage-2 candidate annotations without being mixed into v0.

### Relationship To Existing Token CE

`token_ce` remains valid for Channel-A and for non-trie training surfaces. `stage2_trie_ce` is the Channel-B objective for trie-enabled Stage-2 configs.

For Channel-B, `token_ce` and `stage2_trie_ce` are mutually exclusive unless a future explicit advanced mode defines a non-overlapping composition. This avoids silently training both a single repaired path and a multi-positive trie over the same positions.

### Candidate Aggregation

For each original image example, Stage-2 compiles one merged trie over all valid K rollout-derived repaired candidates. It must not average K independent one-path losses. A merged trie is what gives the model a multi-positive next-token target at branch points.

Source and branch weights encode evidence source in the sidecar metadata:

- valid rollout candidate;
- fallback candidate;
- matched GT object;
- recovered FN object;
- neutral FP context;
- weak-positive FP context.

Stage-2 trie CE v0 consumes those alternatives as hard CE candidates only. It
keeps `support_weight`, `balance_weight`, and semantic role weight keys in the
config contract as reserved future-facing knobs, but non-default values are not
active in this implementation slice.

### False-Negative Insertion

The implementation keeps three insertion policies:

- `tail_append`: append recovered false negatives after matched predictions.
- `fn_slot_shuffle`: insert recovered false negatives into randomized slots among supervised objects, controlled by run seed.
- `sorted`: deterministic clean control, retained because earlier evidence suggests it may harm generalization.

Main ablations should compare `tail_append` and `fn_slot_shuffle`. `sorted` exists as a diagnostic control, not as the favored default.

### False-Positive Policy

The v0 false-positive policies are:

- `zero_loss_context`: keep unmatched predicted objects in the serialized context when needed, but assign zero object/token loss and exclude them from positive trie children.
- `weak_positive_context`: if an unmatched prediction is supported by explorer rollouts, give it weak object-entry encouragement without desc, coord, or geometry supervision.

Weak-positive policy defaults:

```yaml
stage2_ab:
  channel_b:
    fp_policy:
      mode: weak_positive_context
      weak_positive_weight: 0.05
      require_explorer_support: true
      min_support_count: 1
      require_token_score: false
```

Token/span scores are collected in v0 but do not gate false-positive inclusion. This preserves evidence for future threshold filters without prematurely baking in a noisy heuristic.

### Fallback Supervision

Invalid or empty rollout fallback uses `fallback_gt_fn_append_only` corrective supervision. It participates in trie training with a downweighted loss multiplier, but it is not valid rollout evidence.

Default:

```yaml
stage2_ab:
  channel_b:
    fallback_loss_weight: 0.25
```

Optional bootstrap ablation:

```yaml
stage2_ab:
  channel_b:
    fallback_loss_weight: 0.5
```

Fallback candidates must not count as:

- rollout success;
- pseudo-positive support;
- false-positive support;
- healthy compact-full parsing evidence.

A run is considered fallback-dominated when fallback loss share exceeds 0.35 over the tiny-overfit window.

### Normalization

Stage-2 trie CE v0 uses `token_mean` normalization only.

Semantic image bucket balancing is reserved for a future objective refinement. The v0 implementation should fail fast if a config requests `semantic_image_bucket_balanced`, because no active code path currently balances structural, description, coordinate, EOS, fallback, or weak-positive buckets separately.

### Precision

Loss internals that compute log-probabilities, support balancing, and denominator aggregation should use float32-safe math, following the Stage-1 recursive detection loss precedent. The model forward can remain bf16/fp16 according to the training config. The objective boundary must make dtype conversions explicit.

### Packing

Stage-2 trie support should work with the current packed or segmented teacher-forcing metadata. Candidate target positions are per segment before compilation and must be projected to batch-row positions through the existing segment-view helpers. If a config produces unsupported packed metadata, the failure should occur before a model step with a clear error naming the unsupported metadata shape.

### Monitoring

The implementation must preserve training-loop clarity by emitting structured diagnostics outside the core model forward path.

Required scalar families for v0:

- `loss/B/stage2_trie_ce`
- `stage2_trie/target_positions`
- `stage2_trie/branch_points`
- `stage2_trie/max_branching_factor`
- `stage2_trie/candidate_count_mean`
- `stage2_trie/fallback_candidate_share`
- `stage2_trie/fallback_loss_share` as a temporary compatibility alias for the candidate-share proxy
- `stage2_trie/fallback_dominance_warning`
- `stage2_trie/fp_policy_weak_positive_count`

Role-split losses, semantic bucket denominators, and true fallback loss-share
accounting are reserved for a future continuous/objective refinement once the
hard CE path has tiny-overfit evidence.

Required sidecar diagnostics:

- `monitor_dumps/stage2_trie_span_scores/step_<global_step>.jsonl`

Each JSONL record should include:

- `sample_id`
- `rollout_index`
- `candidate_source`
- `span_role`
- `object_role`
- `token_start`
- `token_end`
- `mean_token_logprob`
- `min_token_logprob`
- `object_iou`
- `support_count`
- `loss_weight`

Span roles should include at least:

- `matched_clean`
- `inserted_fn`
- `recovered_fn`
- `neutral_fp`
- `weak_positive_fp`
- `fallback_fn`

## Config Contract

A Stage-2 trie config should express the objective through the existing objective pipeline:

```yaml
stage2_ab:
  channel_b:
    insertion_order: fn_slot_shuffle
    fallback_loss_weight: 0.25
    triage_posterior:
      num_rollouts: 4
    fp_policy:
      mode: weak_positive_context
      weak_positive_weight: 0.05
      require_explorer_support: true
      min_support_count: 1
      require_token_score: false
  pipeline:
    objective:
      - name: token_ce
        enabled: true
        weight: 1.0
        channels: [A]
        application:
          preset: anchor_text_only
      - name: stage2_trie_ce
        enabled: true
        weight: 1.0
        channels: [B]
        application:
          preset: rollout_trie_hard_ce
        config:
          support_weight: 1.0
          balance_weight: 1.0
          struct_weight: 1.0
          desc_weight: 1.0
          coord_hard_ce_weight: 1.0
          eos_weight: 1.0
          normalization: token_mean
```

The schema should reject `token_ce` and `stage2_trie_ce` both targeting Channel-B in the same config.
It should also reject `normalization: semantic_image_bucket_balanced` and non-default reserved weight values for Stage-2 trie CE v0.

## Expected Validation

The first validation target is tiny train-side memorization, not eval. The question is whether Stage-2 can learn or recover the training samples when the training loop is given enough direct, rollout-aware multi-positive supervision.

Minimum train8 target:

- recall greater than or equal to 0.95;
- fraction of images with recall equal to 1.0 greater than or equal to 0.875;
- invalid and empty rollout rates near zero or decreasing;
- fallback loss share below 0.35;
- no increase in duplicate burst or invalid sequence counters.

Minimum train64 target:

- recall greater than or equal to 0.85;
- fraction of images with recall equal to 1.0 greater than or equal to 0.70;
- fallback loss share below 0.35;
- invalid and empty rollout rates not increasing.

Comparison baseline:

- beat the current Stage-2 single-path CE tiny train64 baseline recorded during this investigation: recall about 0.728, recall-equals-one fraction about 0.406, and F1 about 0.715, without improving recall by simply increasing false positives, duplicates, or invalid fallbacks.

## Risks

### Risk: Multi-positive Trie Masks Real Rollout Failure

Mitigation: fallback candidates are downweighted, separate from valid rollout candidates, and excluded from rollout success metrics. Raw rollout artifacts remain materialized.

### Risk: Weak False Positives Teach Unlabeled Noise

Mitigation: weak-positive policy requires explorer support and uses small weight. Token/span scores are diagnostic-only in v0, so the system can compare policy behavior before adding score thresholds.

### Risk: Objective Double Counts Channel-B

Mitigation: schema validation rejects simultaneous Channel-B `token_ce` and `stage2_trie_ce`.

### Risk: Dtype Instability

Mitigation: Stage-2 trie loss reuses Stage-1 recursive CE float32 internal math patterns while preserving model-forward precision.

### Risk: Ordering Ablation Becomes Hidden Data Augmentation

Mitigation: insertion policy is config-owned, run-seeded, and recorded in provenance plus metrics.

## Implementation Entry Points

Primary files expected to change:

- `src/config/schema.py`
- `src/trainers/teacher_forcing/module_registry.py`
- `src/trainers/teacher_forcing/objective_pipeline.py`
- `src/trainers/teacher_forcing/modules/stage2_trie_ce.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/trie_supervision.py`
- `src/trainers/stage2_two_channel/types.py`
- `tests/test_stage2_trie_supervision.py`
- `tests/test_stage2_trie_ce_module.py`
- `tests/test_stage2_ab_config_contract.py`
- `tests/test_stage2_ab_training.py`

Stage-1 reference files:

- `src/detection/objective.py`
- `src/detection/loss.py`
- `src/trainers/metrics/recursive_detection.py`

The implementation should reuse concepts from Stage-1 recursive detection CE, but it should not directly mix the Stage-1 trainer mixin into the Stage-2 trainer. Stage-2 already owns its loss path through `run_stage2_objective_pipelines`.
