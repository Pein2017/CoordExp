## Context

The ablation compares the current sorted object-instance policy against a random-order policy for dense-object training in two places:

- stage-1 SFT training
- stage-2 two-channel training with `channel-A only`

Today, object instance ordering affects the full pipeline:

1. data -> dataset ordering and prompt construction
2. transforms/packing -> stage-1 static packing and encoded-cache eligibility
3. training/inference -> teacher-forced targets, canonical prefixes, rollout/eval prompt rebuilding, and standalone inference messages
4. artifacts -> resolved config and summary metadata used to interpret experiment results

The current system already supports `custom.object_ordering: sorted|random`, but the relevant surfaces are not fully aligned for this ablation:

- stage-1 base dataset random ordering is epoch-sensitive, while static packing currently preserves a fixed plan and does not propagate epoch changes through the packed wrapper
- random ordering is intentionally ineligible for encoded-sample cache reuse
- stage-2 Channel-A training inherits object order into teacher-forced payloads and canonical prefixes
- trainer-driven rollout/eval prompt rebuilding can honor ordering, but standalone inference still hard-codes sorted prompt text

The design therefore needs to preserve experimental comparability while staying within the project constraints:

- YAML-first configuration; no new CLI flags
- preserve Qwen3-VL chat-template compatibility
- preserve geometry semantics and `do_resize=false`
- do not modify upstream HF model internals
- keep the change reproducible and paper-ready

## Goals / Non-Goals

**Goals:**
- Define one explicit ordering contract across dataset fetch, prompt text, stage-2 A-path serialization, rollout/eval prompt rebuilding, and standalone inference.
- Treat `random` as per-epoch reshuffle semantics for this ablation.
- Preserve stage-1 packing parity by extending static packing only where epoch propagation is required for length-invariant reordered samples.
- Keep the ablation config-first, with explicit YAML surfaces for training and inference ordering.
- Make cache behavior explicit so the ablation arms do not silently differ in reuse policy.
- Add verification that catches prompt/order drift across training, rollout/eval, inference, and artifacts.

**Non-Goals:**
- Changing geometry encoding, coord-token semantics, or resize behavior.
- Modifying Channel-B matching/loss behavior beyond shared rollout/eval prompt alignment.
- Making random-order datasets eligible for encoded-sample caching.
- Rewriting static packing into a per-epoch dynamic pack-plan system.
- Adding new CLI flags or patching upstream HF/Qwen internals.

## Decisions

### 1. Ordering remains config-first, with distinct training and inference YAML surfaces

Training continues to use `custom.object_ordering`. Standalone inference gains `infer.object_ordering` so offline inference can request the same sorted/random policy without reusing training-only config namespaces.

`random` is defined as:

- reshuffle object instance order each epoch
- deterministic for a fixed seed, sample identity, and epoch
- independent of `object_field_order`

Why this choice:

- The user explicitly chose reshuffle-each-epoch semantics.
- Keeping inference under `infer.*` matches the current config organization and avoids introducing CLI-only controls.
- Leaving field order and instance order independent preserves the existing contract boundary.

Alternatives considered:

- Fixed-per-run random ordering: rejected because it does not test the requested per-epoch sensitivity hypothesis.
- Reusing `custom.object_ordering` inside inference configs: rejected because inference already has an `infer.*` configuration surface and should remain self-contained.

### 2. Dense prompt wording stays centrally resolved and becomes the single source of truth for sorted/random parity

Prompt text for training, rollout/eval rebuilding, and standalone inference will all resolve through the same ordering-aware dense prompt builder. The ordering contract and prompt wording remain owned by the shared prompt and ordering specs, while trainer-specific specs only describe where those resolved prompts are applied.

Resolved inference artifacts will record the selected ordering alongside prompt metadata so that evaluation output remains auditable.

Why this choice:

- The ablation is specifically about prefix sensitivity, so prompt wording cannot drift from object ordering.
- Centralizing resolution avoids separate sorted/random wording logic in inference and trainer helper code.
- Recording resolved ordering closes the loop for reproducibility audits.

Alternatives considered:

- Leaving prompt ownership split between prompt specs and inference-engine specs: rejected because it duplicates one contract across multiple spec owners.
- Allowing rollout/eval prompt rebuilding to remain an implementation detail: rejected because prompt-prefix alignment is already a contract-critical surface.

### 3. Stage-1 static packing preserves deterministic pack plans and only adds epoch propagation for length-invariant reordered samples

The design does not replace static packing with per-epoch plan regeneration. Instead, it keeps the existing deterministic raw/aligned plan behavior and narrows the change to the minimum needed for this ablation:

- propagate epoch changes through the static packed dataset path
- allow per-fetch object-order reshuffle when the encoded sample length remains invariant under reordering
- preserve existing deterministic pack ordering, DDP alignment, and pack-count semantics

This means static packing remains valid only for datasets where reordered samples keep the same packed-length contract. If length invariance does not hold, the system must fail fast rather than silently mixing packing semantics across ablation arms.

Why this choice:

- Preserving packing parity was the chosen tradeoff.
- Rebuilding pack plans each epoch would change the packing regime and confound the comparison.
- The current blocker is epoch propagation in the packed wrapper, not the entire static packing model.

Alternatives considered:

- Disable packing for the ablation: rejected because throughput/memory changes would become part of the comparison.
- Recompute the static plan each epoch: rejected because it violates the current deterministic static-plan contract and introduces a broader change than needed.

### 4. Cache parity is handled explicitly in ablation configs, not by changing cache eligibility rules

Random-order runs remain cache-ineligible under the existing cache contract. To avoid mixed experimental conditions, the ablation configs will make cache policy explicit instead of relying on defaults. The simplest default is to disable encoded-sample cache in both sorted and random ablation arms.

Why this choice:

- Cache eligibility is already intentionally restricted to deterministic encoded outputs.
- Making random runs cache-eligible would change the cache contract and undermine reproducibility guarantees.
- Config-level parity is enough to keep the ablation fair without expanding system complexity.

Alternatives considered:

- Teach the cache to handle epoch-varying random ordering: rejected because encoded payload is no longer a stable pure function of base record plus fixed config.
- Leave cache policy implicit: rejected because the sorted arm could keep a cache-only advantage while the random arm bypasses or errors.

### 5. Validation must explicitly cover random-order parity across all affected surfaces

The design requires targeted verification in four places:

- prompt parity: sorted vs random wording across training and inference
- stage-1 packing parity: epoch propagation under static packing with deterministic behavior preserved
- stage-2 parity: Channel-A teacher-forced target/prefix ordering and trainer-eval prompt rebuilding
- artifact parity: resolved ordering recorded in inference outputs and summaries

Why this choice:

- The failure mode here is not a crash; it is a silent mismatch between surfaces.
- The study depends on being able to attribute result differences to ordering, not to hidden prompt or caching drift.

Alternatives considered:

- Rely only on existing sorted-order tests: rejected because those tests do not exercise the random-order contract the ablation introduces.

## Risks / Trade-offs

- [Random reordering changes token length for some samples] -> Keep static packing restricted to length-invariant reordered samples, probe/verify the invariant, and fail fast when it does not hold.
- [Inference and training ordering drift again over time] -> Route both through the same prompt resolver, add ordering metadata to artifacts, and add explicit parity tests.
- [Stage-2 A-only and rollout/eval use different prompt text] -> Keep rollout/eval prompt rebuilding under the shared rollout contract and verify random-order parity in trainer-eval tests.
- [Ablation arms differ because only one side uses encoded cache] -> Make cache policy explicit in the ablation configs, with the default design assumption that both arms disable encoded-sample cache.
- [Epoch-sensitive randomness becomes non-deterministic across workers/ranks] -> Continue deriving random ordering from stable sample identity plus epoch and preserve existing determinism tests around multiworker fetches.

## Migration Plan

1. Land the OpenSpec delta specs for prompt ownership, ordering semantics, packing behavior, stage-2 A-path behavior, and rollout/eval prompt alignment.
2. Implement schema/config support for `infer.object_ordering` and artifact metadata emission.
3. Implement the stage-1 packed-dataset epoch propagation change and keep fail-fast guards for non-length-invariant cases.
4. Wire the ordering contract through standalone inference, stage-2 Channel-A, and rollout/eval prompt rebuilding.
5. Add ablation configs for sorted vs random arms with explicit cache policy.
6. Run targeted tests for prompt parity, inference metadata, static packing epoch propagation, and stage-2 rollout/eval parity before implementation is considered complete.

## Open Questions

- None for the architectural direction. The remaining work is specification and implementation detail, not a missing design decision.
