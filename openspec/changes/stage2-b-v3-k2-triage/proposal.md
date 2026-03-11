## Why

Stage-2 Channel-B currently follows the landed v2 clean-prefix contract:

- run one rollout,
- strict-parse and bbox-filter it,
- sequentially dedup same-desc near-duplicates,
- match the clean accepted set against GT,
- build one clean teacher-forced target,
- append FN GT objects,
- apply `duplicate_ul` only to duplicate-certified continuations.

That was the right minimal correction for the raw-prefix duplication failure mode, but the latest evidence shows the residual problem is broader than same-desc duplicate cleanup:

- many baseline misses are recoverable under small-K stochastic decoding,
- many apparent false positives are plausibly unlabeled or class-ambiguous real regions,
- crowded-scene failures often look like coarse same-region re-descriptions or repeated nearby proposals rather than simple clone spam,
- and the hottest rollout behavior still contains explorer-only junk that should not become a second teacher trajectory.

We need a Stage-2 Channel-B contract that uses **two rollout views** to distinguish:

- GT-backed object hypotheses,
- stable-but-unlabeled-consistent anchor hypotheses that should remain neutral context,
- and dead anchor-side continuations that should be removed and mildly suppressed,

while remaining a **minimal perturbation of v2**:

- anchor remains the teacher trajectory,
- one merged clean-prefix forward remains the training shape,
- recovered GTs stay on the FN-injection path,
- and no explorer-prefix distillation or pseudo-label promotion lands in v1.

## What Changes

- Replace the current duplicate-only Channel-B correction rule with a canonical **K=2 triage** contract:
  - anchor rollout = greedy / deterministic,
  - explorer rollout = stochastic, config-driven, default `temperature=0.7`.
- Keep the existing per-run v2 spine for **both** rollouts:
  - bounded salvage + strict record acceptance,
  - bbox-valid filtering,
  - sequential dedup into `accepted_objects_clean`,
  - Hungarian matching under the existing Channel-B gating contract.
- Add a cross-rollout **geometry-first triage** stage over accepted objects:
  - deterministic one-to-one max-IoU anchor/explorer association with stable tie-breaks,
  - side-specific training actions:
    - `anchor_gt_backed`
    - `recovered_fn`
    - `shielded_anchor`
    - `dead_anchor`
- Build the final positive target as an **anchor-edited clean sequence**:
  - keep anchor GT-backed objects,
  - keep anchor shielded objects as neutral context,
  - remove anchor dead objects,
  - append GT FN objects, with higher desc+geo+coord weight for `anchor-missed / explorer-recovered` GTs.
- Keep training operationally simple:
  - one merged teacher-forced forward on the single edited target,
  - no separate explore teacher-forced pass,
  - no recovered-prefix distillation in v1.
- Reuse the existing `duplicate_ul` module slot and logging key for the local negative signal, but broaden its semantic source from “duplicate-certified continuation” to **dead anchor-side continuation** chosen by the new triage stage.
- Add typed config for the explorer rollout and v3 triage thresholds under `stage2_ab.channel_b`.
- Require a per-call rollout decode-override seam so both HF and vLLM backends can issue greedy anchor and stochastic explorer requests in the same B step.
- Add triage metrics, monitor-dump payloads, and docs/spec updates so the new behavior is auditable and reproducible.

## Capabilities

### Modified Capabilities
- `stage2-ab-training`: replace single-rollout duplicate-only Channel-B correction with the canonical K=2 anchor-edited triage contract.
- `rollout-matching-sft`: add per-call decode overrides for dual-policy Channel-B rollouts across HF and vLLM backends.
- `teacher-forcing-objective-pipeline`: keep `duplicate_ul` as the canonical B-only objective module but broaden its prerequisite metadata from duplicate-only to dead-anchor continuation targets.
- `teacher-forcing-unified-loss-registry`: define `gt_backed`, `shielded`, `dead_anchor`, and `recovered_fn` rollout semantics under one merged clean-prefix forward.
- `trainer-metrics-components`: add triage counts plus aggregation-safe numerators/denominators and clarify that `loss/B_rollout_text/duplicate_ul` now covers dead anchor-side continuation suppression, not only same-desc duplicate cleanup.

## Impact

- Stage-1 behavior is unchanged.
- Channel-A remains unchanged in objective shape and scheduling intent.
- Stage-2 Channel-B becomes a deliberate contract update:
  - anchor remains the only teacher trajectory,
  - explorer is a miner, not a teacher,
  - explorer-only non-GT-backed objects are dead by default in v1,
  - recovered GTs are weighted through FN injection rather than distilling the explorer prefix,
  - recovered weighting applies to desc+geo+coord terms and therefore requires new per-object rollout metadata and module changes.
- Channel-B compute/latency on B steps increases because each sample now requires two rollout calls through the same backend topology, and both HF and vLLM rollout paths must support per-call decode overrides.
- Existing duplicate diagnostics remain useful, but they become supporting evidence rather than the whole worldview.
- Docs/spec/config updates become part of the done-definition because this change alters the canonical Stage-2 training contract.
