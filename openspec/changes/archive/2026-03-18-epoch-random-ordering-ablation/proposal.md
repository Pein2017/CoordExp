## Why

Current dense-object training uses a sorted object sequence that may help convergence during SFT, but it also bakes a strong prefix-order prior into the prompt and teacher-forced target. For the planned ablation between the current sorted policy and a per-epoch-random policy, we need training, packing, stage-2 A-only prefix construction, rollout-driven evaluation, and standalone inference to follow one consistent ordering contract; otherwise the study will compare infrastructure drift instead of object-order sensitivity.

## What Changes

- Add a config-first ablation path for `sorted` versus `random`, where `random` explicitly means reshuffling object instance order each epoch for stage-1 and stage-2 two-channel `channel-A only`, with prompt wording coupled to the selected ordering policy.
- Preserve packing parity for the ablation by extending stage-1 static packing only as needed for length-invariant object-order reshuffle, so epoch propagation is honored without discarding deterministic pack-plan behavior or DDP stability.
- Align stage-2 A-only teacher-forced payloads and canonical clean-prefix construction with the configured object-ordering policy, and keep trainer-driven rollout/eval prompt rebuilding on the same ordering-aware contract.
- Add explicit standalone inference support for ordering-aware dense prompts through `infer.object_ordering`, and record the resolved ordering in inference artifacts alongside prompt metadata so offline evaluation remains reproducible.
- Make encoded-sample cache policy explicit for the ablation configs so the arms do not silently differ in cache behavior; random-order runs remain cache-ineligible, and sorted-order ablation configs must not receive an unplanned cache-only advantage.
- Keep the change YAML-first and reproducibility-focused: no new CLI flags, explicit config coverage for the ablation arms, and validation/tests that make order-contract drift fail fast.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `dataset-prompt-variants`: extend the dense prompt contract so sorted/random ordering policy remains explicitly coupled to prompt text across training, rollout/eval, and standalone inference, including `infer.object_ordering` and resolved-ordering artifact metadata.
- `object-field-ordering`: update the shared ownership boundary so object instance ordering remains governed by `custom.object_ordering`/`infer.object_ordering`, and `random` is explicitly defined as per-epoch reshuffle semantics while field-order behavior remains independent.
- `packing-dataset`: change the static packing contract so stage-1 can preserve packing parity while honoring length-invariant epoch-wise object-order reshuffle deterministically.
- `stage2-ab-training`: update the stage-2 two-channel Channel-A contract so teacher-forced targets and canonical prefixes honor the configured object instance ordering without changing unrelated rollout matching semantics.
- `rollout-matching-sft`: update the shared rollout/eval prompt-encoding contract so trainer-driven evaluation prompt rebuilding uses the same ordering-aware dense prompt resolution as training.

## Impact

Affected areas include stage-1 and stage-2 experiment configs, dense prompt resolution, shared ordering config/schema behavior, dataset ordering and static packing, stage-2 A-path prefix serialization, rollout/eval prompt rebuilding, standalone inference message construction, and ablation-specific cache settings. Expected code touchpoints include `configs/stage1/`, `configs/stage2_two_channel/`, `src/config/prompts.py`, `src/config/schema.py`, `src/config/loader.py`, `src/datasets/dense_caption.py`, `src/datasets/wrappers/packed_caption.py`, `src/trainers/stage2_two_channel.py`, `src/trainers/stage2_rollout_aligned.py`, `src/infer/engine.py`, and the associated OpenSpec/docs/tests that guard reproducibility and eval validity.
