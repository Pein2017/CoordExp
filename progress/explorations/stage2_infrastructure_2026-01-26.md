---
title: Stage-2 Infrastructure Exploration Request
status: historical-exploration
scope: stage2
kind: exploration
topics: [stage2, infrastructure, rollout, trainer-integration, historical-design]
references:
  - docs/PROJECT_CONTEXT.md
  - docs/SYSTEM_OVERVIEW.md
  - docs/IMPLEMENTATION_MAP.md
---

# Stage-2 Infrastructure Exploration Request

Date: 2026-01-26
Last updated: 2026-01-26
Note: referenced run artifacts and external checkouts may be pruned; paths are best-effort pointers.

Status note:
- this is a historical exploration request from before the current Stage-2 stack was implemented,
- it is useful for understanding early requirements and design pressures,
- the current live Stage-2 entrypoints are `src/sft.py`, `src/trainers/stage2_two_channel.py`, and `configs/stage2_two_channel/`.

## Context

I have completed **Stage-1 training** (see `progress/pretrain/stage1_foundation.md`), which implements standard SFT with coord token distribution losses (softCE + W1 + gate). The model now reliably outputs structured detection format with `<|coord_k|>` tokens.

## Historical goal at the time

Proceed to **Stage-2: EM-ish rollout training** as specified in `progress/directions/stage2_emish_set_supervision_v1.md`. This required infrastructure that supports:

1. **Rollout loop**: Autoregressive generation (greedy/beam) to produce model predictions
2. **Forward pass on rollout context**: Teacher-forced forward pass using the model's own rollout tokens (self-context)
3. **Flexible loss computation**: Geometric losses (L1/GIoU) computed from CoordExp-decoded continuous coordinates, plus standard CE on reordered GT sequences
4. **Efficient batching**: Handle rollout → matching → forward → loss computation in a batched training loop

## Exploration Task

**Comprehensively explore** the following packages (installed in `ms` conda environment or available locally) to identify existing infrastructure that can support Stage-2 rollout/forward loops:

- **transformers** (installed in the active `ms` conda environment)
- **ms-swift** (local upstream checkout or installed package; see `docs/standards/UPSTREAM.md`)
- **vLLM** (if installed in `ms` environment)

### Key Requirements

The infrastructure should provide:

1. **Rollout/generation utilities**:
   - Autoregressive generation with configurable decoding (greedy/beam/temperature)
   - Ability to capture intermediate logits during generation (if needed)
   - Batch-friendly generation APIs

2. **Forward pass control**:
   - Teacher forcing on custom token sequences (rollout tokens, not just GT)
   - Access to logits at specific positions (coord token positions)
   - Gradient computation control (enable/disable gradients selectively)

3. **Training loop integration**:
   - Compatible with ms-swift trainer infrastructure (or clear integration path)
   - Support for custom loss computation hooks
   - Efficient memory management for rollout → forward sequences

4. **Flexibility**:
   - Minimal assumptions about output format (we use structured `<obj>`/`<desc>`/`<box>` protocol)
   - Configurable via YAML (preferred) or minimal code adapters

### Historical success criteria

- Identify concrete classes/functions/modules that can be reused or adapted
- Document integration points with existing CoordExp code (`src/sft.py`, `src/metrics/dataset_metrics.py`)
- Propose a minimal implementation plan that leverages existing infrastructure rather than manual implementation
- Flag any gaps that require custom implementation

## Historical expected deliverables

1. **Exploration report**: Summary of relevant APIs in transformers/ms-swift/vLLM with code references
2. **Integration plan**: How to wire rollout → matching → forward → loss into the training loop
3. **Minimal prototype path**: Suggested approach for a Stage-2 training entry point (for example, a hypothetical `src/sft_stage2.py` or trainer mixin)

## References

- Stage-1 spec: `progress/pretrain/stage1_foundation.md`
- Stage-2 design: `progress/directions/stage2_emish_set_supervision_v1.md` (sections 6-8)
- Current training entry: `src/sft.py`
- Loss mixins: `src/metrics/dataset_metrics.py`

For current code navigation, prefer `docs/SYSTEM_OVERVIEW.md` and `docs/IMPLEMENTATION_MAP.md` over this archived request.
