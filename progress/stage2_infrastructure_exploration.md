# Stage-2 Infrastructure Exploration Request

## Context

I have completed **Stage-1 training** (see `progress/pretrain/first_stage.md`), which implements standard SFT with coord token distribution losses (softCE + W1 + gate). The model now reliably outputs structured detection format with `<|coord_k|>` tokens.

## Goal

Proceed to **Stage-2: EM-ish rollout training** as specified in `progress/full_idea.md`. This requires a training infrastructure that supports:

1. **Rollout loop**: Autoregressive generation (greedy/beam) to produce model predictions
2. **Forward pass on rollout context**: Teacher-forced forward pass using the model's own rollout tokens (self-context)
3. **Flexible loss computation**: Geometric losses (L1/GIoU) computed from CoordExp-decoded continuous coordinates, plus standard CE on reordered GT sequences
4. **Efficient batching**: Handle rollout → matching → forward → loss computation in a batched training loop

## Exploration Task

**Comprehensively explore** the following packages (installed in `ms` conda environment or available locally) to identify existing infrastructure that can support Stage-2 rollout/forward loops:

- **transformers** (`/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`)
- **ms-swift** (`/data/home/xiaoyan/AIteam/data/ms-swift`)
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

### Success Criteria

- Identify concrete classes/functions/modules that can be reused or adapted
- Document integration points with existing CoordExp code (`src/sft.py`, `src/metrics/dataset_metrics.py`)
- Propose a minimal implementation plan that leverages existing infrastructure rather than manual implementation
- Flag any gaps that require custom implementation

## Expected Deliverables

1. **Exploration report**: Summary of relevant APIs in transformers/ms-swift/vLLM with code references
2. **Integration plan**: How to wire rollout → matching → forward → loss into the training loop
3. **Minimal prototype path**: Suggested approach for a Stage-2 training entry point (e.g., `src/sft_stage2.py` or trainer mixin)

## References

- Stage-1 spec: `progress/pretrain/first_stage.md`
- Stage-2 design: `progress/full_idea.md` (sections 6-8)
- Current training entry: `src/sft.py`
- Loss mixins: `src/metrics/dataset_metrics.py`
