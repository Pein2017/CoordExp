# Human-13 All-HF Shared-Surface Trajectory Credit Design

## Ownership

The scientific owner is
[`unit.md`](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/unit.md).
The implementation authority is
[`add-human13-all-hf-shared-surface-trajectory-credit-vertical`](../../../openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/).
This document is an execution-oriented design summary and does not restate or
override their requirements.

## Selected design

Build one experiment-local HF session around a single live BF16/FA2 Source
model.  It performs K16 stochastic acquisition in four no-cache, full-history
groups of four and then replays each completed group in one grad-enabled
teacher-forced forward on the same model object.  The replay tensors are
admitted only after every chosen-token history and processed log probability
passes the frozen parity contract.

Once admitted, reuse the existing trajectory-credit ledger, sparse compiler,
fresh-AdamW actual-delta reconstruction, owner-wise preservation projection,
private checkpoint evaluator, and complete training-state rollback.  The first
live experiment runs the full combined algorithm once on image 1584 and audits
clean greedy at RP 1.0 and RP 1.10.  It does not spend the first result on a
large ablation matrix.

## Why this approach

Three approaches were considered:

1. **Selected: shared HF BF16/FA2 sampling and replay.**  It sacrifices
   sampling throughput, keeps backward feasible, and directly removes the
   cross-engine mismatch that prevented the algorithm from running.
2. **Rejected for this probe: fp32/SDPA throughout.**  It offers a familiar
   audit surface but makes the simultaneous model, optimizer, gradients, and
   replay state likely to exceed one-card memory.
3. **Deferred: vLLM sampling with an explicitly approximate/off-policy
   estimator.**  It is the likely scale path if the algorithm works, but first
   requires evidence that the algorithm is worth scaling.

The sampler deliberately disables KV/prefix reuse.  The replay remains
vectorized because per-token full-history backward would make one 1,500-token
K16 group prohibitively slow.  Token-level parity owns this one remaining
physical-shape difference.

## Interfaces

The new shared-surface module owns four public concepts:

```text
SharedSurfaceIdentity
HFSharedSurfacePlan
SampledHFGroup
GradientReplayGroup
```

One `HFSharedSurfaceSession` constructs and admits these values.  Constructors,
artifact loaders, publishers, and consumers all call the same admission seam.
No downstream module receives a loose model/tokenizer pair.

The one-image runtime consumes admitted replay groups and existing frozen
manifest/alias artifacts, then produces:

```text
trajectory numerator
compiler numerator
actual AdamW delta
projected delta
dual-RP owner outcome
rollback reproduction
terminal disposition
```

## Execution topology

- GPU 0 owns BF16/FA2 sampling, replay, backward, AdamW, and preservation.
- GPU 1 owns the existing fp32/SDPA batch-one Source/proposal clean-greedy
  audit.
- The live entry defaults to zero-action dry-run and requires explicit model/GPU
  authority plus two distinct cards.
- Private proposal bytes exist only inside the audit transaction and are never
  promoted.

## Behavioral decision

The one-image update is scientifically complete after one admitted update,
both audits, and exact rollback, regardless of whether the outcome is good.
Only a protected positive result may expand to the 13-image panel:

- RP 1.0 gains at least one K-hit/Source-miss H owner;
- RP 1.0 net unique owners is positive;
- both RP surfaces lose zero Source G owners; and
- duplicates do not increase, malformed output remains absent, and neither
  audit hits the token cap.

## Failure behavior

- Surface/parity failure: zero update; typed implementation HOLD.
- Missing objective/projection component: zero private apply; no fallback CE.
- Completed but null/unsafe proposal: publish result, rollback, stop width
  expansion.
- Rollback mismatch: durable failure; no later execution.

## Self-review

- No placeholders or unresolved scientific choices remain.
- Sampling and replay share model state but not physical sequence shape; the
  parity gate explicitly owns that distinction.
- The design tests the complete algorithm before adding ablation breadth.
- The predecessor result remains immutable and is not reinterpreted.
- No model/GPU execution is authorized by this document.
