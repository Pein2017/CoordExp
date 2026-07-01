## Why

Stage-2 (two-channel) and rollout-aligned training currently have a fragmented and partially duplicated loss/weight surface across trainer code, module helpers, config schema, and pipeline manifest defaults. This makes it difficult to audit “effective” loss weights, and it has already produced objective drift (e.g., `text_gate_weight` is configured but currently a no-op).

This change is needed now to (1) implement `text_gate` to actively prevent coord-token probability mass in text positions, and (2) remove backwards-compat aliases so the loss system has a single maintainable contract.

## What Changes

- Implement `text_gate` as a real `coord_reg` sub-term that penalizes coord-vocab mass at `type=struct|desc` (text) positions, respecting rollout-context FP-neutral masking and EOS rules. **BREAKING**
- Refactor and regroup loss terms and weight assignment into a single unified registry + objective pipeline contract, so “effective weights” are derived from one place. **BREAKING**
- Remove all backwards-compat loss weight aliases and legacy config keys (e.g., duplicate `*_weight` spellings, legacy per-trainer fallbacks). **BREAKING**
- Enforce strict (fail-fast) per-module config validation for both:
  - `stage2_ab.pipeline.*.config` (Stage-2 two-channel), and
  - `rollout_matching.pipeline.*.config` (rollout-aligned SFT).
  **BREAKING**
- Reduce metric key sprawl by emitting only canonical `loss/<component>` keys for registry-defined loss components; remove trainer-specific `loss/*_anchor` / `loss/*_self_context` keys. **BREAKING**
- Reduce monitor spam by sparse-emitting rollout-only keys (e.g., `rollout/precision|recall|f1`) only on steps where rollout actually executed (avoid constant `0.0` monitors in Channel-A-only runs). **BREAKING**

## Capabilities

### New Capabilities
- `teacher-forcing-unified-loss-registry`: Canonical loss component registry (contexts, token types, masking semantics) shared across Stage-2 two-channel and rollout-aligned training, including `text_gate` math/semantics.

### Modified Capabilities
- `stage2-ab-training`: Update requirements to depend on the unified loss registry for module definitions, strict config validation, and canonical metric key emission (no legacy aliases).
- `rollout-matching-sft`: Update requirements to strictly validate pipeline module configs and to use the same unified loss registry (including `text_gate`) for rollout-context supervision.
- `trainer-metrics-components`: Allow breaking removal of legacy loss metric keys in favor of canonical registry-derived keys.

## Impact

- **Training configs**: YAML profiles under `configs/stage2_two_channel/**` and rollout-matching configs will need updates to the new canonical weight keys (no aliases).
- **Config schema**: Typed schema and strict parsing will be tightened; typos that previously no-op’d will fail fast.
- **Trainer code**: Stage-2 two-channel and rollout-aligned trainers will share the same loss module implementations and weight resolution; duplicated weight plumbing will be removed.
- **Metrics/logging**: Some legacy metric keys will be removed; downstream dashboards/scripts must migrate to canonical `loss/<component>` keys.
