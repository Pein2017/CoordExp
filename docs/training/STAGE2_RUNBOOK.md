# Stage-2 Training Runbook (Deprecated Legacy Notes Rewritten)

This runbook was rewritten after the removal of Channel-A self-context
iteration. The historical iteration-specific notes are no longer active
training guidance.

## Current Supported Contract

- `custom.trainer_variant: stage2_two_channel`
- Channel-A runs a single GT-anchored teacher-forced forward.
- Channel-B keeps the rollout-aligned clean-prefix path.
- Channel-B pseudo-positive mode is opt-in through:
  - `stage2_ab.channel_b.pseudo_positive.enabled`
  - `stage2_ab.channel_b.pseudo_positive.coord_weight`
- Supported objective presets:
  - `token_ce.application.preset: anchor_text_only`
  - `loss_dead_anchor_suppression.application.preset: rollout_only`
  - `bbox_geo.application.preset: anchor_only`
  - `bbox_size_aux.application.preset: anchor_only`
  - `coord_reg.application.preset: anchor_only`
- Pseudo-positive mode keeps the one-forward contract:
  - `matched_clean` -> coord + matched-prefix structure CE
  - `fn_injection` -> coord + FN desc CE
  - `pseudo_positive` -> coord only
  - `shielded_anchor` -> no positive supervision
  - `dead_anchor` -> no positive supervision, with duplicate-like branch suppression only
- Default authored pseudo-positive profile:
  - `triage_posterior.num_rollouts: 4`
  - `1` anchor + `3` explorers
  - enabled `K=2` remains the explicit no-promotion control
- Enabled failure semantics:
  - malformed anchor preparation drops that sample from Channel-B training
  - malformed explorer preparation aborts the step
  - zero-object explorers remain valid zero-support evidence
- Deprecated authored knobs now fail fast in active/training configs:
  - `stage2_ab.n_softctx_iter`
  - `stage2_ab.softctx_grad_mode`
  - `stage2_ab.softctx_temperature`
  - `stage2_ab.coord_ctx_embed_mode`
  - `stage2_ab.coord_decode_mode`
  - `rollout_matching.coord_decode_mode`

## Recommended Config Entry Points

- A-only baseline: `configs/stage2_two_channel/prod/a_only.yaml`
- Mixed A/B baseline: `configs/stage2_two_channel/prod/ab_mixed.yaml`
- COCO1024 B-majority continuation: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority.yaml`
- Pseudo-positive `K=4` production profile: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
- Production-like smoke: `configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml`
- Pseudo-positive smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml`

## Smoke Commands

Use the repo-standard environment:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/a_only.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml
```

## First Pseudo-Positive Checks

For the first enabled runs, verify:

- `stage2/raw_rollouts` reflects `1 + (K-1)` rollout execution
- `train/triage/pseudo_positive_selected_count` is non-zero on at least some dense scenes
- `train/triage/unlabeled_consistent_count` remains the total shielded-anchor count
- `rollout/explorer/*` remains interpretable as mean-over-valid-explorer-view aggregates
- duplicate-like dead-anchor suppression remains narrow; do not expect all dead anchors to emit suppression targets

## Historical Rationale

The removal decision is documented in:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`

Use that artifact for historical analysis, not as a source of active training
guidance.
