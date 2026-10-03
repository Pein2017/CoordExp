## Why

The resident research vLLM engine already refreshes language DoRA and untied token deltas in place, but exposes only greedy generation and conflates pre-normalization and policy likelihoods. HF rollout acquisition dominates the current training loop, so the shared resident path needs faithful sampled actions and current-policy traces before it can replace that acquisition work.

## What Changes

- Share coordinate median-normalization arithmetic between differentiable HF replay and resident vLLM generation, preserving current arithmetic and gradient semantics.
- Add seeded, full-support temperature-one sampling to the resident research API while retaining existing greedy defaults.
- Capture both pre-normalization and normalized selected-action likelihoods from the actual vLLM generation, including EOS and valid PAD actions, with bounded storage and explicit request/snapshot association.
- Qualify synchronous HF update, in-place vLLM refresh, cache invalidation, acknowledgement, and next-version generation through the existing runtime probe; measure the complete acquisition and refresh cost.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-infras-research-probe-infra-base`: resident continuation sampling, paired action likelihoods, and acknowledged current-version refresh.

## Impact

Changes are limited to maintained research mechanics in `src/qwen`, their probe consumer, focused tests, and the existing technical qualification entrypoint. Production scored-inference contracts, training objectives, checkpoint formats, installed vLLM sources, and the active frozen A/B run remain unchanged. Implementation and outputs live in an isolated temporary worktree until safe integration. No new scientific arm or quality claim is introduced.
