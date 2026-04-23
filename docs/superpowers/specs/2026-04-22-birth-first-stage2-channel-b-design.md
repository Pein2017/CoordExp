---
title: Birth-First Stage-2 Channel-B Design
date: 2026-04-22
status: draft
change: birth-first-stage2-channel-b
---

# Birth-First Stage-2 Channel-B Design

## Problem

The current Stage-2 Channel-B path is already clean-prefix and one-forward, but the active training signal still leans more toward anchor-hypothesis refinement than object birth under rollout. Recent diagnostics also suggest that some misses are tied to local `EOS` preference rather than pure visual incapacity. The next design therefore needs to improve recall without turning unmatched predictions into broad pseudo-label positives or making duplicate suppression the primary lever.

This design is intentionally scoped to one fixed base-model plus adapter pair:

- base model:
  `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- adapter checkpoint:
  `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`

## Approaches Considered

### 1. Stronger duplicate suppression first

Pros:
- simplest story if duplicate bursts were the main failure mode
- preserves current recall machinery

Cons:
- recent diagnostics do not support it as the main next lever
- risks deleting healthy crowded multiplicity without improving object birth

Verdict:
- rejected as the primary design

### 2. Broader pseudo-positive / multi-view promotion

Pros:
- could widen recall support more aggressively
- uses more explorer evidence

Cons:
- partial-annotation policy gets entangled with object-birth policy
- harder to interpret in the first real decision round
- increases support-aggregation complexity immediately

Verdict:
- deferred until after a simpler birth-first decision round

### 3. Birth-first `K=2` clean-prefix contract

Pros:
- keeps anchor-first, one-forward, clean-prefix structure
- makes recall and stop calibration the center of the change
- easier to audit and compare against the baseline

Cons:
- narrower than the eventual multi-view path
- may leave some recall on the table if `K>2` support later matters

Verdict:
- recommended

## Recommended Design

Use an opt-in birth-first Channel-B mode with:

- `1` anchor rollout + `1` explorer rollout
- support-positive retained anchors as structure-first positives
- recovered GT objects kept on the weighted FN path
- one local continue-over-EOS margin per recovered boundary, projected through the existing rollout-text surface
- unchanged duplicate-burst unlikelihood as a narrow guardrail

The central behavior change is:

```text
from: coord-corrective retained-anchor training
to:   birth-first retained-anchor + recovered-boundary training
```

## Why This Is The Right Next Step

- It is the smallest change that directly targets recall.
- It does not require a new model head, RL loop, or second teacher-forced pass.
- It gives a clean discriminative training round before spending large compute on a long run.
- It keeps the study checkpoint fixed so the contract, not the checkpoint family, is what changes.

## Main Risks

- support-positive prefix structure could reinforce some unlabeled false regions;
- continue-over-EOS could increase sequence length if it is not kept local;
- `K=2` may be too narrow for the final best-performing recipe.

## Mitigations

- keep support-positive retained anchors structure-only
- keep continue-over-EOS limited to recovered-GT boundaries
- treat `K=2` as the decision profile first, not the permanent final answer

## Expected Review Questions

- Is the support-positive bucket too permissive?
- Should a follow-up isolate a pure stop-first `K=2` variant before widening the contract further?
- Should the long run stay `K=2` if the decision round wins cleanly, with any `K>2` path handled in a separate follow-up?
