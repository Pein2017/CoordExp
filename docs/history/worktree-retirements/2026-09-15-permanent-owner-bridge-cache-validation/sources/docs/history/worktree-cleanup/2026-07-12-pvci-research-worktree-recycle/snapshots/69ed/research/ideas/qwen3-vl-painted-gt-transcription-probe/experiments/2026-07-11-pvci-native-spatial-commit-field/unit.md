---
title: PVCI Native Spatial Commit Field
description: Distinguishes an instance- or region-level native commit field from exact coordinate-token memory and fixed geo-sorted continuation.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-native-spatial-commit-field
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - autoregressive-commit
  - causal-counterfactual
updated: 2026-07-11
---

# PVCI Native Spatial Commit Field

## Question

After an otherwise symmetric canonical row commits same-class instance `A` or
`B` on the same clean image, what does frozen Qwen3-VL suppress at the next row
boundary?

This unit asks whether the native text prefix carries a spatial/object-specific
commit field, only ordinary memory of exact coordinate token IDs, a class-wide
inhibition signal, or a fixed `geo_sorted` continuation state. It is a causal
same-image prefix intervention. It does not propose a ledger architecture.

## Why This Is the Next Decider

The preceding native-commit unit found immediate raw-logit suppression of the
committed row and increased scores for annotated-uncovered rows, but all strict
free successes were also compatible with the learned `geo_sorted` successor.
The same-class coordinate span carried most of the observed separation. Those
facts leave two very different mechanisms observationally equivalent:

1. a useful native instance/region commit substrate that later training could
   strengthen; or
2. ordinary autoregressive memory for recently emitted coordinate tokens plus
   raster/list continuation.

The smallest way to separate them is to hold image, class phrase, row count,
syntax, and prefix length fixed while swapping only which spatial instance was
committed, then test exact and nearby non-token-overlapping geometries.

## Functional Requirement, Mechanism Hypothesis, Experimental Handle

- Functional requirement: `commit(A)` must reduce the chance of selecting `A`
  again while leaving distinct same-class instance `B` available, and the
  effect must swap when `B` is committed.
- Mechanism hypothesis: a native prefix-induced state represents the committed
  visual support or instance beyond verbatim coordinate token repetition.
- Experimental handle: symmetric canonical GT rows for same-class `A` and `B`
  on the same clean image, with direct post-prefix scoring and one-row free
  continuation.
- Candidate implementation if supported: strengthen or make writable this
  native commit state before adding a new slot/register abstraction.

The handle is not the proposed implementation. GT rows are privileged causal
interventions used only to identify native decoder semantics.

## Competing Hypotheses

### H1: Spatially generalized native prefix field

Committing `A` selectively suppresses exact `A` and a local family of nearby
`A` geometries, while committing `B` swaps that field to `B`. The effect remains
when the nearby candidate shares zero coordinate token IDs with the committed
row and does not simply follow the frozen geo successor. This distinguishes a
spatially generalized prefix state from verbatim token memory; it does not by
itself distinguish visually grounded instance memory from smooth coordinate-
token representations.

### H2: Exact coordinate-token memory

Only the four emitted coordinate token IDs, or candidates reusing them, are
suppressed. Spatially nearby boxes with zero shared coordinate token IDs do not
inherit the effect. Offline repetition penalty may amplify this pattern, but a
native spatial field is absent.

### H3: Class-wide inhibition

Because `A` and `B` share the same phrase, both same-class rows are suppressed
similarly regardless of committed geometry. Spatial swap contrasts are weak.

### H4: Fixed traversal or row-position state

The learned `geo_sorted` successor or a fixed spatial direction wins after
either commit. The specifically suppressed/gaining instance does not swap with
the counterfactual committed row. Against the shared prompt-only baseline, the
on-policy first object `A` may suppress itself after `commit_A`, while the
deliberately off-order `B` promotes itself after `commit_B`; this asymmetric
transition means prefix state is acting as an implicit raster cursor rather
than an order-free commit ledger.

### H5: Diagnostic score state without behavioral readout

Raw candidate scores swap spatially, but one-row free generation does not avoid
the committed support or continues to STOP/malformed output. Native information
exists but ordinary decoding fails to read it robustly.

### H6: Spatial revisit attractor rather than commit suppression

The most recent row positively primes the same geometry and nearby zero-token-
overlap geometries. Swapping `commit_A` to `commit_B` swaps an attractive field
from A to B: `Q(A) < 0` and `Q(B) > 0`. Free decoding preferentially repeats the
just-committed pair member. This would expose a native spatial persistence
mechanism with the wrong sign for coverage, offering a direct mechanism-level
explanation for duplication bursts.

## Frozen Panel and Score-Blind Selection

- Source universe: the already frozen 24 first-to-second primary events in the
  prior unit's `selected_transitions.jsonl`; no score or outcome from the prior
  scorer participates in event selection.
- Eligibility: `A` has at least one annotated-uncovered object with the same
  normalized description; `B` must not be the frozen next `geo_sorted`
  successor; prefer `IoU(A,B) <= 0.10`.
- Deterministic `B`: maximum normalized center distance from `A`, then GT index
  as a fixed tie break.
- Initial gate: two eligible events chosen by stable event ID order.
- Bounded panel: all eligible events, expected to be approximately five based
  on score-blind ledger inspection. Do not expand because an effect is weak.
- COCO caveat: unlabeled space is `unknown/unannotated`, never verified
  background. No global visible-object coverage claim is permitted.

## Prefix Conditions

For every event, both branches have the identical clean image, ordinary task
prompt, one completed row, canonical syntax, same phrase, and equal semantic
role:

- `prompt_only`: no committed row; measurement baseline only.
- `commit_A`: canonical phrase of `A` plus canonical GT coordinates of `A`.
- `commit_B`: the same phrase plus canonical GT coordinates of `B`.

The headline contrast is `commit_A` versus `commit_B`, not either diagnostic
branch versus the uninterrupted native source rollout. The native emitted `A`
prefix may be retained only as a separately labeled comparability control.

## Candidate Geometry Panel

Each prefix scores the same candidate rows with the shared class phrase:

- `exact_A`, `exact_B`;
- `near_A_zero_token_overlap`, `near_B_zero_token_overlap`: high-IoU spatial
  perturbations with none of the four coordinate token IDs equal to the source;
- `mid_A_zero_token_overlap`, `mid_B_zero_token_overlap`: moderate-IoU
  perturbations, again with zero shared coordinate token IDs;
- optionally one deterministic `unknown_unannotated_control` that has low
  overlap with every labeled object, reported only as unknown/unannotated.

Perturbations are created in normalized coordinate-bin space with deterministic
search and explicit achieved IoU/token-overlap receipts. A candidate that cannot
meet its predeclared constraints is retained as infeasible with a reason; it is
not silently replaced after scoring.

## Primary Measurements

All model execution remains at the checkpoint's native dtype (`bfloat16`).
Measurement logits and log-softmax are cast to `float32`; sequence sums use
Python double precision and bucket log-sum-exp uses `float64`.

For candidate `x` after commit branch `c`, record exact teacher-forced sequence
log likelihood and span decomposition:

```text
S_c(x) = log P(row_x | image, prompt, commit_c)
```

Primary swap contrast:

```text
Q(x) = S_commit_B(x) - S_commit_A(x)
```

H1 predicts `Q(A-family) > 0` and `Q(B-family) < 0`: `A` becomes less
suppressed when the commit swaps away from `A`, while `B` becomes more
suppressed when the commit swaps onto `B`. Report raw logits as primary and
offline RP1.1 separately. Never combine them into one headline.

Also record:

- exact-token versus zero-token-overlap spatial generalization;
- local-field decay from exact to near to mid candidates;
- phrase, coordinate, and structural span contributions;
- frozen geo-successor identity and whether the free result follows it;
- STOP as a separate one-token diagnostic, not a conserved probability bucket;
- one-row free continuations at RP1.0 and RP1.1, parsed and attributed to
  `A`, `B`, other annotated objects, unknown/ambiguous, malformed, or STOP.

## Causal Controls

- Same-class `A/B` holds the phrase fixed and isolates spatial identity.
- Symmetric canonical rows prevent a native-versus-GT prefix asymmetry.
- Equal completed-row count and canonical syntax control generic list state.
- Zero coordinate-token overlap separates spatial locality from verbatim token
  repetition.
- `B` is not the frozen geo successor, placing committed identity in conflict
  with raster continuation.
- Prompt-only scores expose generic one-row continuation changes but are not
  substituted for the paired causal swap.

Phrase/geometry cross-splices for different-class objects are deferred. They
answer a different binding question and would add unnecessary factors before
the spatial-versus-token distinction is settled.

## Gates and Outcome Map

### Runtime gate

Before the bounded panel, require on two events:

- exact checkpoint, config, tokenizer, prompt, image-plan, and special-token
  identities;
- exact tokenize/decode round trip for both canonical commit rows;
- equal row lengths where the shared phrase makes this possible, with any
  difference explicitly receipted;
- finite float32 candidate logits and deterministic repeated scoring;
- code, unit, config, checkpoint, ledger, argv, Git HEAD/branch/dirty, and
  output hashes written before interpretation.

### H1-supporting result

- the sign of the exact A/B swap contrast follows the committed instance on a
  majority of eligible events;
- the same sign extends to at least the near zero-token-overlap candidates;
- it is present in raw scores, not created only by RP1.1;
- and at least one free-decode policy shows a matching behavioral tendency
  without a compensating collapse into STOP/malformed output.

This is evidence for a trainable native spatial substrate, not proof that the
state is visually object-bound and not architecture promotion. A positive
result requires a later off-object or cross-image causal control before the
word `instance ledger` is justified.

### H2-supporting result

- exact A/B rows swap, but zero-token-overlap near/mid candidates lose the
  effect, especially in raw scores. Next work should treat explicit spatial
  commitment as missing rather than claiming a native region ledger.

### H3/H4-supporting result

- both same-class instances move together, or the same geo successor wins after
  either branch; or
- relative to the shared prompt baseline, `commit_A` suppresses exact/near A
  while the counterfactual off-order `commit_B` promotes exact/near B on a
  strict majority of all attempted events.

Next work should test whether order diversification or an order-free objective
can make commitment symmetric before introducing explicit memory.

### H5-supporting result

- diagnostic scores swap but free continuation does not. The bottleneck is
downstream readout/continuation/STOP, and a score-only architecture claim is
prohibited.

### H6-supporting result

- exact and near zero-token-overlap candidates show the reverse of the desired
  suppression swap on a strict majority of all frozen attempted events; and
- under at least one fixed decode policy, both A and B branches repeat their
  own committed pair member on a strict majority without denominator filtering.

This supports a native spatial revisit attractor, not a commit ledger. The next
training question becomes whether a commit/coverage objective can invert or
redistribute this existing write/read state while preserving recognition.

## Stop Conditions

Stop this unit when the two-event runtime gate plus all frozen eligible events
have explicit outcomes, or earlier if the runtime/evidence contract fails.
Do not add slots, a detector, a trained renderer, new loss, or broad training.
Do not add more mark styles, attention maps, ordering sweeps, or outcome-selected
samples. A result that does not distinguish H1-H5 closes as inconclusive and
must name the next single discriminator.

## Result

The bounded five-event panel completed with `5/5` scored events, no retained
contract failures, exact repeated per-token scores, native `bfloat16` model
execution, and float32-or-higher measurement. Architecture promotion remains
`not_promoted`.

The desired symmetric commit signature was absent:

- raw exact desired suppression swap: `0/5`;
- raw near zero-token-overlap desired suppression swap: `0/5`;
- raw mid zero-token-overlap desired suppression swap: `0/5`;
- no free continuation selected the other A/B pair member under RP1.0 or
  RP1.1.

The shared prompt-only baseline identifies a stronger alternative:

- exact: canonical on-policy `commit_A` suppresses A while counterfactual
  off-order `commit_B` promotes B in `5/5` events;
- near zero-token-overlap: the same asymmetry holds in `4/5` events;
- mid zero-token-overlap: it holds in `3/5` events;
- median raw exact `ΔA(A)=-11.2735`, while `ΔB(B)=+8.9273`.

Therefore this checkpoint does not expose an order-free native commit-to-
uncovered ledger. The identified result is narrower: a descriptive asymmetry
between canonical first/on-policy A and a far, later/off-order B forced into the
same first row slot. It remains compatible with a raster traversal state,
generic first-row/off-policy-prefix behavior, object size/location priors,
class priming, and smooth coordinate-token representations. Zero-token-overlap
local generalization rules out only verbatim coordinate-token repetition.

The immediate route-deciding question is whether the sign follows identity or
row slot/order. The smallest discriminator is a same-image identity-by-slot
counterbalance on this same checkpoint: compare `A→B` and `B→A` adjacent
transitions, score the actual geo successor explicitly, and retain the exact,
near, and mid geometry controls. Only after this removes the complete
A-first/B-later confound should an order-diversified continuation checkpoint be
trained. If the counterbalanced effect follows slot/order, test order
diversification before adding explicit memory; if it remains identity-specific
at both slots, investigate visual/object coupling next.

Canonical artifacts:

- scorer: `/data/CoordExp/outputs/painted_gt/pvci_native_spatial_commit_field/full5_v1/`;
- analysis: `/data/CoordExp/outputs/painted_gt/pvci_native_spatial_commit_field/full5_v1/analysis_v4/`;
- frozen plan: `/data/CoordExp/outputs/painted_gt/pvci_native_spatial_commit_field/plan_pure_ce_step4887/`.

## Evidence Scope

- Checkout: `/data/CoordExp/.codex/worktrees/69ed/CoordExp`.
- Checkpoint: long-trained pure-CE/type-gated step-4887 from the preceding unit.
- Source rollout: completed clean val200 RP1.1 artifact from the preceding unit.
- Frozen input ledger:
  `/data/CoordExp/outputs/painted_gt/pvci_native_commit_uncovered_redistribution/plan_pure_ce_step4887/selected_transitions.jsonl`.
- Planned output root:
  `/data/CoordExp/outputs/painted_gt/pvci_native_spatial_commit_field/`.
- Authority: non-normative research evidence only.

## Completion Promise

The unit completes with a durable plan, per-event scorer outputs, analysis JSONL
and summary/report, full denominators including failures/infeasible controls,
precision and code-identity receipts, focused tests, and one direct verdict:
`spatial suppressive prefix field`, `spatial revisit attractor`, `exact-token
memory`, `class-wide shift`, `fixed traversal`, `score-only state`, or
`inconclusive`. Architecture remains
`not_promoted` regardless of a positive diagnostic result.

## Research Unit Closeout

Observed: The exact five-event panel completed. Canonical first/on-policy A
suppressed itself, while the forced far/off-order B promoted itself; the
direction generalized to zero-token-overlap nearby geometry but was not
symmetric across the pair.

Supported: A bounded first/on-policy versus later/off-order prefix asymmetry in
the frozen checkpoint. The prefix changes coordinate-row probabilities beyond
verbatim token reuse.

Not supported yet: A visually grounded object commit field, order-free ledger,
persistent coverage, or detection benefit.

Next decider: The completed identity-by-slot counterbalance, followed by the
planned canonical-depth probe after its inconclusive result.

Promotion decision: `not_promoted`; no architecture was promoted.
