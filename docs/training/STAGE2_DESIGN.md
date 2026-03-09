---
doc_id: docs.training.stage2-design
layer: docs
doc_type: design
status: canonical
domain: training
summary: Stable design overview for the current Stage-2 training path.
updated: 2026-03-09
---

# Full Idea v2

This page is the stable training-design overview for the landed clean-prefix Stage-2 Channel-B direction.

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
  - clean-prefix teacher forcing plus duplicate unlikelihood.

The key v2 change is that Channel-B no longer treats the raw rollout prefix as the positive teacher-forced prefix.

Instead, the canonical path is:

```text
raw rollout
  -> bounded container salvage + strict record acceptance
  -> bbox-valid filtering
  -> sequential dedup
  -> accepted_objects_clean + duplicate bursts by boundary
  -> Hungarian on accepted_objects_clean
  -> clean-prefix teacher forcing + duplicate_ul
```

## Why v2 Exists

The old raw-prefix Channel-B path let later correct objects inherit duplicate-heavy prefixes.

That made it too easy for the model to:
- keep one salvageable matched localization,
- keep producing near-duplicate objects,
- preserve some recall,
- but worsen prediction count, duplicate density, and truncation.

The v2 fix is intentionally narrow:

- keep generic unmatched clean extras neutral by default,
- remove duplicate-certified objects from the positive prefix,
- attach them to clean boundaries,
- apply unlikelihood only at the first true LCP-divergence token.

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
  - `src/trainers/teacher_forcing/modules/duplicate_ul.py`

## Key Config Surfaces

- `stage2_ab.pipeline`
- `stage2_ab.channel_b`
- `rollout_matching.*`
- `configs/stage2_two_channel/`

## Detailed Historical Writeup

The full long-form design narrative remains available here:

- [progress/directions/stage2_clean_prefix_v2_longform.md](../../progress/directions/stage2_clean_prefix_v2_longform.md)

Use that file when you want the extended derivation, examples, and historical argumentation.
