# Proposal Bridge Closeout and the Evidence-First Workflow Reset

Source session: 019f4a19-d81c-75a2-84b0-2c20379e686e, principally Jul 10 to Jul 14 2026.

## Why this chapter matters

The first major research branch made a technically real intervention and a
useful negative scientific finding. It also exposed a workflow failure that
changed how later research units were designed. Future agents should not
rebuild the bridge or repeat its long rollout panel merely because held-out
representation probes looked positive.

## The bridge hypothesis

The early diagnosis was that pure cross-entropy might produce conservative,
short autoregressive rollout because it lacked a durable, causally consumed
object commitment state. The proposed A/B/C study was deliberately narrow:

* A was the continuation baseline;
* B added a proposal auxiliary objective without feedback into row generation;
* C added the same proposal objective plus a proposal-conditioned residual
  bridge into row generation.

It did not authorize slots, a persistent ledger, coverage repulsion, an
external detector, or terminal hacks. Repetition penalty 1.10 was treated as
an inherited anti-duplication heuristic to ablate, not as a mechanism.

## What was verified before the scientific failure

The implementation was not dismissed because of an obvious wiring bug. It
passed real Qwen-path checks covering no-op parity, causal isolation, packed
segment isolation, cache behavior, arm-neutral cache identity, and
gradient/update ownership under the relevant attention backends. A
deterministic Flash Attention configuration resolved a native gradient-repeat
issue. All arms completed matched 512-step training, and B/C passed the
predeclared held-out representation gate.

This demonstrates a reusable principle:

    representation improvement is not evidence that free rollout causally
    consumes the representation in the intended way.

## The behavior that rejected the bridge

C-on increased continuation and row count but produced high duplication,
invalid output, truncation, and precision loss. The key controls were more
informative than the headline recall change:

* another-image feedback approximately reproduced C-on;
* token permutation remained strongly active;
* position-only feedback was intermediate;
* norm-matched random was near baseline;
* lower repetition penalty made the C-on pathology worse.

Therefore learned structured feature distribution mattered, but correct
image-token correspondence did not. The bridge was a global continuation or
row-prior carrier, not object-specific visual-to-text pairing. The full
evaluator returned a safety non-inferiority hold. It is a closed scientific
route, not a latent deployment candidate.

## Durable engineering lessons

* A no-op operation that changes the backward graph is not a no-op control.
* A float32-owned parameter can still be computed under bfloat16 autocast.
* Real forward/cache-path smoke is more valuable than a probe-only success.
* Loss denominators must be defined over actual supervised positive bags.
* Artifact identity, prompt/template identity, parser behavior, and runtime
  configuration must be validated before behavior is compared.
* A stale shard must never be relabeled and reused as evidence.

## Workflow reset caused by this result

The evidence review did not reject the negative conclusion. It rejected the
costly path used to obtain it: a large build and long rollout matrix preceded
the smallest target-specific causal discriminator. The user and lead adopted a
lighter evidence-first method:

1. State the question, smallest execution path, expected evidence, and
   non-goals.
2. Run the smallest real smoke early.
3. Repair only a defect that changes scientific interpretation.
4. Use curated case studies and visual review before broad infrastructure.
5. Keep experiment-local code thin and reusable only where runtime evidence
   proves the seam is worth retaining.
6. Promote a shared contract or architecture only after the research gate.

This does not mean that receipts, tests, or runtime semantics are optional.
It means they should be proportional to conclusion risk, not a substitute for
the primary observation.

## Where to continue instead

The correct successor was not another bridge variation. It was to identify
whether frozen native prefixes already contain object support, whether coherent
phrase-geometry rows change successor state, and which route differences are
safe enough to shape greedy behavior. The core reading path is the dense
enumeration compass and the sampled-rescue research units.

Formal handles:

* research/investigations/qwen3-vl-dense-enumeration/compass.md
* research/investigations/qwen3-vl-dense-enumeration/overview.md
* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/
