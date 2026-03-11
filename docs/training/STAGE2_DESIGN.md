---
doc_id: docs.training.stage2-design
layer: docs
doc_type: design
status: canonical
domain: training
summary: Stable design overview for the current Stage-2 training path.
updated: 2026-03-09
---

# Full Idea v3

This page is the stable training-design overview for the landed K=2 triage Stage-2 Channel-B direction.

Use it to understand the design intent before dropping into the runbook or the main specs.

For exact normative behavior, still prefer:
- [Stage-2 AB spec](../../openspec/specs/stage2-ab-training/spec.md)
- [Teacher-forcing objective pipeline spec](../../openspec/specs/teacher-forcing-objective-pipeline/spec.md)
- [Unified loss registry spec](../../openspec/specs/teacher-forcing-unified-loss-registry/spec.md)

## Core Idea

CoordExp keeps Stage-2 split into two complementary channels:

- **Channel-A**:
  - hot path,
  - expectation / self-context stabilization,
  - cheap and frequent.

- **Channel-B**:
  - cold path,
  - rollout-time set/cardinality correction,
  - dual-rollout triage over anchor/explorer hypotheses,
  - one merged clean-prefix teacher-forced forward plus dead-anchor suppression.

The key v3 change is that Channel-B no longer reasons about a single rollout only.

Instead, the canonical path is:

```text
anchor rollout + explorer rollout
  -> per-run bounded salvage + strict record acceptance
  -> per-run bbox-valid filtering
  -> per-run sequential dedup
  -> per-run accepted_objects_clean + Hungarian matching
  -> deterministic anchor/explorer association
  -> triage: anchor_gt_backed | recovered_fn | shielded_anchor | dead_anchor | dead_explorer
  -> anchor-edited clean target + weighted FN injection + dead-anchor loss_dead_anchor_suppression
```

## Why v3 Exists

The clean-prefix v2 correction removed a large raw-prefix duplication failure mode, but the remaining problem is broader than same-desc duplicate cleanup.

The v3 contract addresses four residual cases together:
- baseline misses that become recoverable under one stochastic explorer rollout,
- stable unlabeled-consistent anchor objects that should stay neutral rather than be punished,
- dead anchor continuations that should be removed and mildly suppressed,
- crowded-scene overlap patterns that are broader than exact-desc duplicate bursts.

The resulting v1 implementation is intentionally conservative:
- anchor remains the teacher trajectory,
- explorer is a miner, not a second teacher,
- explorer-only non-GT-backed objects are dead by default,
- recovered GT objects are weighted through FN injection rather than via recovered-prefix distillation.

## Stable v1 Contract

- Anchor rollout is greedy / deterministic.
- Explorer rollout is stochastic under `stage2_ab.channel_b.triage_posterior`.
- The final target is built by editing the anchor clean sequence, preserving anchor order.
- Shielded anchor objects remain neutral context only.
- Dead anchor objects are removed from the positive prefix and sourced into `loss_dead_anchor_suppression`.
- Recovered GTs stay on the FN tail and receive higher per-object desc+geo+coord weight.
- Training remains one merged teacher-forced forward on the edited target.

## What Stays Stable

- no DETR-style detection head
- no objectness classifier
- Stage-1 remains the language-of-boxes foundation
- Channel-A remains the hot stabilizer path
- geometry still uses CoordExp expectation decode with SmoothL1 + CIoU

## Current Recommended Reading

After this page, read:

1. [Stage-2 Runbook](STAGE2_RUNBOOK.md)
2. [Metrics & Losses](METRICS.md)
3. the relevant OpenSpec pages

## Key Code Handles

If you are moving from design to implementation, these are the main code handles:

- trainer entrypoint:
  - `src/sft.py`
- main Stage-2 two-channel trainer:
  - `src/trainers/stage2_two_channel.py`
- rollout parsing and matching helpers:
  - `src/trainers/rollout_matching/parsing.py`
  - `src/trainers/rollout_matching/matching.py`
- teacher-forcing pipeline registry / atoms:
  - `src/trainers/teacher_forcing/module_registry.py`
  - `src/trainers/teacher_forcing/objective_atoms.py`
- explicit duplicate UL module:
  - `src/trainers/teacher_forcing/modules/loss_dead_anchor_suppression.py`

## Key Config Surfaces

- `stage2_ab.pipeline`
- `stage2_ab.channel_b`
- `stage2_ab.channel_b.triage_posterior.*`
- `rollout_matching.*`
- `configs/stage2_two_channel/`

## Detailed Historical Writeup

The full long-form design narrative remains available here:

- [progress/directions/stage2_clean_prefix_v2_longform.md](../../progress/directions/stage2_clean_prefix_v2_longform.md)

Use that file when you want the extended derivation, examples, and historical argumentation.
