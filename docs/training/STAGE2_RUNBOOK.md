# Stage-2 Training Runbook (Deprecated Legacy Notes Rewritten)

This runbook was rewritten after the removal of Channel-A self-context
iteration. The historical iteration-specific notes are no longer active
training guidance.

## Current Supported Contract

- `custom.trainer_variant: stage2_two_channel`
- Channel-A runs a single GT-anchored teacher-forced forward.
- Channel-B keeps the rollout-aligned clean-prefix path.
- Supported objective presets:
  - `token_ce.application.preset: anchor_text_only`
  - `loss_dead_anchor_suppression.application.preset: rollout_only`
  - `bbox_geo.application.preset: anchor_only`
  - `bbox_size_aux.application.preset: anchor_only`
  - `coord_reg.application.preset: anchor_only`
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
- Production-like smoke: `configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml`

## Smoke Commands

Use the repo-standard environment:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/a_only.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml
```

## Historical Rationale

The removal decision is documented in:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`

Use that artifact for historical analysis, not as a source of active training
guidance.
