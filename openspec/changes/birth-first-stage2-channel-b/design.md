## Context

The current Stage-2 two-channel contract is already anchor-first, clean-prefix, and one-forward, which are all properties worth preserving. The problem is not that Channel-B lacks machinery; it is that the currently active training pressure still leans toward geometry correction of anchor-retained objects more than toward reliable object birth under rollout.

The repo history backing this change is consistent on three points:

- the current pseudo-positive path can train stably, but it has often behaved more like coord-correction than count or structure expansion;
- duplicate-burst unlikelihood is useful as a narrow guardrail, but it has not been the dominant active learning signal in the recent windows that were inspected;
- some false negatives are not well described as pure perceptual misses, because short `EOS now` continuations can still beat plausible object continuations on total sequence score.

This change is therefore a contract rewrite for the next decision round, not a broad architecture reset. It is also a fixed base-model plus adapter study: every design choice here assumes the base model stays `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` and the only adapter checkpoint in scope is `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`.

## Goals / Non-Goals

**Goals:**
- Preserve the current anchor-first, clean-prefix, one-forward Stage-2 shape.
- Make missing-object birth the main Channel-B positive surface.
- Add a minimal and explicit stop/continue correction for recovered-GT boundaries.
- Keep duplicate control narrow and local rather than broadening unmatched-negative policy.
- Keep the decision profile interpretable enough to justify a large real-training budget after one short decision round.

**Non-Goals:**
- Introduce a new detector head, reward model, or RL loop.
- Turn explorer-only non-GT-backed objects into unrestricted prefix positives.
- Make `K=4` or arbitrary-`K` support mandatory for the first birth-first contract.
- Replace the current geometry pipeline, output format, or Qwen3-VL template contract.
- Change the checkpoint family under study.

## Decisions

### 1. Keep the canonical Stage-2 shape and reuse the existing `K=2` control profile

The new contract stays anchor-first, clean-prefix, and one-forward. The key simplification is that birth-first mode reuses the existing `K=2` anchor-plus-one-explorer profile as the study control surface rather than widening support aggregation at the same time.

Why this over keeping `K=4` as the primary research profile:
- `K=2` isolates whether the gain comes from better birth and stop calibration rather than from larger support aggregation.
- it keeps the association, retained-prefix, and recovered-GT logic easier to audit;
- it gives a cheaper and more interpretable discriminative training round before a long run.

Rejected alternative:
- keep `K=4` pseudo-positive as the default decision profile. This was rejected for the first iteration because it would make support aggregation part of the hypothesis under test instead of a later scaling decision.

### 2. Split retained unmatched anchors into support-positive and neutral shielded subsets

Birth-first mode treats retained unmatched anchors as two different semantic buckets:

- `support_positive_shielded`: anchor-retained unmatched objects that have a one-to-one associated explorer object with `IoU >= unlabeled_consistent_iou_threshold` under the current anchor/explorer association contract;
- `neutral_shielded`: retained unmatched anchors with no such associated explorer support.

The important shift is that support-positive retained anchors stop being “mostly geometry correction” state. In birth-first mode they become structure-first positives in the clean prefix while remaining outside extra desc CE and outside positive bbox/coord supervision.

This change does **not** introduce a separate birth-first `cluster_demoted` bucket. Cluster demotion remains part of the non-birth-first pseudo-positive contract only.

Why this over broad pseudo-label promotion:
- it keeps the positive surface anchored to objects the anchor already emitted;
- it avoids turning explorer-only unmatched objects into a noisy pseudo-label union;
- it lets the training signal say “this object should stay live in the ordered prefix” without overclaiming its full semantic correctness.

Rejected alternative:
- promote explorer-only non-GT-backed objects directly into prefix positives. This was rejected for the first iteration because it would collapse partial-annotation policy and object-birth policy into one risky change.

### 3. Reuse global rollout-prefix structure CE instead of adding a second positive text mechanism

Birth-first mode reuses the existing rollout-prefix structure CE surface through `token_ce.config.rollout_global_prefix_struct_ce_weight`. The contract change is behavioral, not architectural:

- support-positive retained anchors MUST participate in that structure surface when birth-first mode is enabled;
- birth-first-enabled configs MUST set `rollout_global_prefix_struct_ce_weight > 0`;
- support-positive retained anchors still create no extra positive desc CE.

Why this over adding new desc-positive or pseudo-caption logic:
- it is the smallest surface that still teaches ordered object birth;
- it preserves current prompt and JSON contracts;
- it does not require a second positive text path just to say “continue this object list.”

Rejected alternative:
- give support-positive retained anchors desc CE. This was rejected because the desc token surface remains the least trustworthy place to hard-promote partially supervised unmatched objects.

### 4. Add a recovered-boundary continue-over-EOS margin as a rollout-text atom

Recovered GT objects remain on the existing FN-injection path and keep the current weighted positive supervision. Birth-first mode adds one more signal: an optional local continue-over-EOS margin at the boundary where the anchor would otherwise stop or choose the wrong next continuation.

This term is:
- boundary-local;
- only defined on recovered-GT boundaries;
- emitted under the existing rollout-text provenance rather than a new authored objective module;
- mean-like across eligible boundaries.

Its exact form is:

- at a recovered boundary `b`, let `c1` be the first canonical continuation token of the recovered object continuation under the final clean target,
- let `s_cont = log p(c1 | b)` and `s_eos = log p(EOS | b)` from the same next-token distribution at boundary `b`,
- with margin `m = stage2_ab.channel_b.birth_first.continue_over_eos_margin`, define:
  `L_continue(b) = max(0, m - (s_cont - s_eos))`,
- the emitted atom is the mean of `L_continue(b)` over eligible recovered boundaries in the optimizer step.

Why this over a broad anti-EOS bias:
- the repo evidence points to local stop pressure, not a universal need to lengthen every sequence;
- a recovered-boundary term is easy to explain and test;
- it keeps the stop-correction signal tied to evidence-backed misses.

Rejected alternatives:
- global EOS downweighting, because it would conflate stop calibration with sequence-length inflation;
- a new standalone objective module, because this change explicitly keeps ownership inside the existing rollout-text projection surface.

### 5. Keep duplicate-burst unlikelihood unchanged as a narrow guardrail

Duplicate-burst unlikelihood remains required in the pipeline and remains the canonical B-only suppressive term. This change does not ask it to solve recall.

Why:
- the recent diagnostics do not support “punish duplicates harder” as the main next move;
- duplicate UL is still valuable for clear dead local continuations;
- keeping it unchanged lets the decision round measure whether birth and stop changes matter on their own.

Rejected alternative:
- make duplicate UL broader or stronger in the same change. This was rejected because it would blur causal attribution during the first real decision round.

### 6. Add explicit birth-first metrics instead of reading everything through duplicate diagnostics

Birth-first mode needs its own observability surface:
- support-positive retained anchor counts,
- neutral retained anchor counts,
- recovered-GT counts,
- continue-over-EOS boundary counts,
- the mean-like continue-over-EOS loss atom.

Why:
- current duplicate-centric monitors are useful but not enough to judge a birth-first contract;
- the decision round needs to tell whether the run improved object birth, stop calibration, or only duplicate appearance.

## Risks / Trade-offs

- **[Support-positive structure CE could reinforce unlabeled false regions]** → Keep the surface structure-only, keep desc CE off, and keep the first contract restricted to anchor-retained objects.
- **[Continue-over-EOS margin could inflate rollout length]** → Restrict it to recovered-GT boundaries only, log boundary counts explicitly, and compare count/length stability against the baseline.
- **[K=2 may underuse multi-view support]** → Accept this for the first decision round; if the contract wins cleanly, `K>2` can be evaluated as a later scaling choice.
- **[Birth-first mode could overlap confusingly with pseudo-positive mode]** → Make the first contract mutually exclusive with pseudo-positive mode and fail fast when both are enabled together.
- **[Fixed-checkpoint scope could overfit the study to one family]** → This is intentional for the current decision round; changing checkpoint family would answer a different question.

## Migration Plan

1. Add the OpenSpec deltas for the birth-first contract, rollout registry semantics, and metrics.
2. Extend typed Stage-2 config validation with `stage2_ab.channel_b.birth_first` while preserving the existing non-birth-first control profile and duplicate-control surfaces.
3. Implement the retained-anchor partition and recovered-boundary continue-over-EOS atom in the current one-forward Channel-B runtime.
4. Add focused schema, target-building, loss-projection, and aggregation tests.
5. Author paired `K=2` control and birth-first smoke/decision configs rooted in the fixed base model `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` plus the fixed adapter checkpoint `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`.
6. Run the short decision round, compare the paired control and birth-first configs under matched runtime conditions, and only then promote the winning `K=2` contract to a long real-training run in this change.

## Open Questions

- Should a follow-up isolate a pure `stop-first K=2` variant with recovered-boundary continue-over-EOS but without support-positive structure supervision, so the next study can separate stop-calibration from structure-positive birth pressure more cleanly?
- If birth-first `K=2` wins, should a later follow-up reintroduce support-weighted coord terms or `K>2` support aggregation, or keep the cleaner control profile as the long-term contract?
