## Context

See [proposal.md](proposal.md) and the owning
[research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-row-contrast-geometry-preservation-successor/unit.md).
The existing Human-13 path already owns sealed manifests, exact token spans,
no-padding multimodal packs, fixed-parameter A4 score/gradient replay,
language-only DoRA assembly, one-rank AdamW training, checkpoint publication,
HF batch-one evaluation, and owner analysis. The successor should extend those
surfaces rather than create a second trainer or generic objective framework.

## Goals / Non-Goals

**Goals:**

- Turn complete duplicate rows into owner-aware negative/positive comparisons.
- Prevent the measured rectangle-order failure at the actual greedy decision
  sites without forcing canonical coordinates.
- Test whether one first-order parameter-gradient constraint reduces `G`
  coordinate drift at the same low dose.
- Retain exact packing, optimizer, checkpoint, and evaluation semantics from
  the predecessor.

**Non-Goals:**

- Online refresh, sequence-level RL, K-miss learning, GT-IoU search, STOP
  supervision, architecture changes, or long-dose optimization.
- A general row-ranking API, generic gradient-surgery trainer, new evidence
  store, or distributed training path.
- Guarantees after AdamW or evidence beyond the same thirteen images.

## Decisions

### 1. A sidecar extends the sealed manifest without rewriting it

An experiment-local successor builder reads the immutable Human-13 manifest
and optional authoritative A4@1/@2 raw outputs. It resolves complete duplicate
row spans, coordinate sites, covered owner state, uncovered `T=G union H`
owners, candidate aliases, and Source-`G` watch rows into a canonical sidecar.

The manifest stays the owner of Source/K discovery. The sidecar owns only new
derived training events and hashes every input artifact. This avoids changing
historical evidence while allowing already-observed post-update duplicate
states to increase negative-state coverage without another decode.

The first low-dose benchmark keeps optional A4@1/@2 states in that sidecar as
diagnostic evidence but trains only on the twelve sealed-manifest duplicate
events. Within each event it freezes one deterministic native alias per
uncovered owner. A real pack census showed that training all optional events
and aliases would require 1.85M packed tokens per exposure; the frozen subset
preserves owner alternatives while keeping this bounded screen near 0.63M.

**Alternative rejected:** online refresh after each successor update. It would
change the data distribution between R1/R2 and violate the fixed, low-cost
contrast.

### 2. Candidate scores are owner-normalized before event contrast

The builder supplies exact teacher-forced candidate rows. The loss helper
computes length-normalized bbox or owner-distinguishing row log scores, then
`logmeanexp`s aliases within owner. Event-level `logsumexp` combines owners.
This lets multiple uncovered owners contribute probability mass without giving
an owner more weight merely because it appeared under more K seeds.

Same-description events exclude shared description/schema tokens and compare
only four coordinates. Cross-description events include the smallest declared
owner-distinguishing description span plus coordinates. When no valid
alternative exists, four stable token-UL terms replace the old y2-only event.

**Alternatives rejected:** full-vocabulary token CE to one chosen row and
whole-row unlikelihood. The first suppresses other valid owners; the second
penalizes shared wrappers and category tokens.

### 3. Rectangle validity is a dynamic set-valued logit constraint

The encoded event records coordinate-token IDs and decoded integer values. At
`x2/y2`, the runner constructs the valid set from the already-emitted left/top
coordinate and takes exact fp32 maxima over valid and complement sets. Selector
indices are detached; gradients flow through the two selected logits through a
hinge that is exactly zero after the frozen margin is satisfied.

This is cheaper and better aligned with greedy decode than enumerating
`1000^4` boxes. It also accepts coordinate aliases that remain metric-usable.

**Alternative rejected:** post-row validity CE or canonical GT coordinate CE.
The former acts after the branch has already failed; the latter changes the
Stage-1 estimand from native-support consolidation to GT coordinate teaching.

### 4. R2 projects the accumulated raw R1 gradient once per exposure

The existing runner already accumulates a complete panel objective at one
parameter state. R2 performs two backward accumulations into separate fp32
trainable-parameter buffers: the full R1 objective and owner/image-normalized
Source-`G` coordinate CE. A small experiment-local helper computes dot products
and norms across the language-DoRA parameters, writes either the unchanged or
projected gradient into `.grad`, then delegates global clipping and AdamW to
the current runtime.

Projecting the complete R1 gradient, including Source replay, makes the
receipt's first-order statement exact. `eps` and numerical tolerance are
sealed in the resolved plan. World-size one avoids a new collective protocol.

**Alternatives rejected:** frozen-source KL and per-microbatch projection. KL
has zero gradient at the initial identical snapshot, while microbatch
projection depends on pack order and does not constrain the accumulated panel
direction.

### 5. Reuse the A4 streaming and live Human-13 entries

R1/R2 use A4's exact two-pass global candidate weighting when a logical image
bundle exceeds one 12k physical pack. Row-contrast, rectangle, replay, and
watch segments are independently no-padding packed and globally normalized;
there is one optimizer mutation per full panel exposure.

The live CLI gains only two successor arm IDs, exposure set `{1,2}`, the
sidecar binding, and R2's projection receipt. Existing model assembly,
checkpoint writer/readback, and eval-matrix code remain canonical.

**Alternative rejected:** a new successor trainer. It would duplicate the
highest-risk runtime seams for no scientific gain.

### 6. A one-image real slice owns launch admission

Image 14038 is fixed before implementation because the predecessor already
contains trusted G/H rows and duplicate behavior on it. The slice uses the
same resolved R1 plan and real model path as the panel run, except for an
explicit one-image projection. It must prove one update, immutable checkpoint
write/read, and HF clean-greedy analyzer compatibility.

After admission, R1 and R2 train on two GPUs. Their four checkpoints are
evaluated independently on four GPUs. This six-GPU topology minimizes the
critical path without adding multi-rank complexity.

## Risks / Trade-offs

- **[Teacher-forced row contrast may not redirect free-running decode]** ->
  Keep original-prompt clean greedy as the only decision-owning readout.
- **[Static A4 negative states are off-policy for R1/R2]** -> Treat them as a
  frozen coverage expansion, report their source separately, and stop at two
  exposures.
- **[Four-coordinate fallback UL can still move probability to bad tokens]** ->
  Use it only with no trusted uncovered alternative and pair it with the
  rectangle-valid gate.
- **[Gradient projection is parameterization- and scale-dependent]** -> Bind
  the exact DoRA trainable surface, normalize watch events, and report norms
  and coefficients rather than claiming invariant preservation.
- **[AdamW can violate the raw-gradient first-order relation]** -> Weight decay
  remains zero; claim only the pre-clip raw gradient property and measure G
  retention empirically.
- **[A4 streaming multiplies forwards]** -> Reuse its existing exact global
  score/replay path and report physical forwards, packed tokens, wall time, and
  peak memory.
- **[Optional prior-output alignment may fail]** -> Fail that optional source
  closed without blocking sealed Source/K events or fabricating spans.

## Migration Plan

This is additive and experiment-local:

1. add sidecar records and pure loss/projection helpers behind CPU tests;
2. extend Human-13 materialization and dry-run for R1/R2;
3. complete the fixed image-14038 real vertical slice;
4. launch R1/R2 to exposures one and two under new immutable roots;
5. evaluate four checkpoints and update the owning research results; and
6. stop, regardless of outcome, until the user chooses a later route.

Rollback is deletion of the successor-only scripts, configs, tests, and loss
helpers. Existing manifests, production defaults, predecessor artifacts, and
historical checkpoints remain unchanged.
