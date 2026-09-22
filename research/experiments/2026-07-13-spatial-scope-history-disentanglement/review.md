---
title: Review Synthesis and Protocol Revision Gate
description: Ranks three independent reviews, records accepted and rejected findings, and defines the fixed-point gate for the revised research unit.
type: investigation
role: review-synthesis
authority: non_normative_research
reviewed_unit: 2026-07-13-spatial-scope-history-disentanglement
status: complete
updated: 2026-07-13
---

# Review Synthesis and Protocol Revision Gate

This file preserves the pre-execution review verdict. The later
[completed research unit](unit.md) and [executed results](results.md) own the
current lifecycle, evidence status, and bounded scientific verdict; the
pre-execution `planned` and `not_authorized` wording below is historical and is
not the current unit status.

## Terminology and Name Registry

- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed
  reportable category set used by the reviewed unit.
- **K spatial calls (`K`)**: the number of canonical cells in the selected
  spatial grid and therefore the number of calls in each multi-call primary
  arm.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: `K` independently
  seeded sampled full-image calls followed by one frozen object merge.
- **Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)**: `K`
  full-size masked canvases exposing one core-plus-halo region each, decoded from
  fresh prompts.
- **Full-Canvas Masked Region with Cumulative Accepted-Row Prefix
  (`MASK_CUMULATIVE`)**: the same masked canvases decoded with rows admitted and
  serialized by the frozen cumulative prompt-state transition.
- **Validation-200 (`val200`)**: the existing fixed 200-image validation scope
  used as the broad five-arm execution scope. Historical outputs on this scope
  remain planning and baseline-sanity evidence only.
- **Secure Hash Algorithm 256-bit (`SHA-256`)**: the content-digest algorithm
  required for immutable ledger and artifact identity.
- **Common Objects in Context crowd flag (`iscrowd`)**: source annotation field
  preserving official crowd-region ignore semantics.

## Scope

This record synthesizes three independent reviews without modifying their source
files:

1. [English-language independent audit](review-English.md)
2. [Claude independent audit](review-claude.md)
3. [Chinese-language independent audit](review-Chinese.md)

The ranking concerns scientific and contract quality, not language, model
identity, or writing style. The criteria are causal discriminative value,
correctness, implementation-contract coverage, actionability, and discipline
about the strongest allowed claim.

## Quality Ranking

### 1. English-language independent audit

This is the strongest overall review. It identifies the most consequential
scientific defect missed by the others: matching the number of calls does not
match the number of sampled opportunities received by each object. Every object
is visible in all full-image bagging calls, while a spatial policy gives the
object one primary owning-cell call. Therefore, spatial restriction exceeding
bagging is a strong policy result, but equality cannot falsify a spatial-scope
benefit.

It also gives the best end-to-end account of input-level masking versus
post-vision localization, sampled-bagging executability, cumulative-prefix
state, object aggregation, manual-ledger provenance, crowd semantics, effect
thresholds, and evidence lifecycle. Its final verdict, **narrow and re-review**,
is the most scientifically appropriate.

Residual weakness: it is long, and some repository or cohort facts were derived
from evidence outside the three named research documents. Those facts are
accepted only as readiness checks to re-execute, not as evidence for this unit.

### 2. Claude independent audit

This is the strongest mechanism and decoding review. It precisely exposes how
prompt-inclusive repetition penalty can differentially affect reset and
cumulative prompts, why a weak-diversity bagging arm is a straw control, how an
object merge can asymmetrically collapse adjacent same-class bagging detections,
and why cumulative native-tile coordinates have no naturally coherent frame.

Residual weaknesses: it does not elevate the per-object opportunity mismatch
to the central causal problem, and its wording that penalties “accumulate” can
be misread. Under the reviewed Transformers implementation, the penalty is
applied once per token identifier present in the current seen-token set; the
set grows with prompt and generated content, rather than the same token being
multiplied repeatedly for every occurrence. Its verdict, **approve after named
corrections**, is too permissive before a focused re-review.

### 3. Chinese-language independent audit

This is the strongest protocol, provenance, and lifecycle review. It clearly
separates the pre-execution image-only audit ledger from post-execution pooled
candidate adjudication, requires an executable cumulative-prefix state
transition, and gives useful artifact, receipt, scoring, and readiness gates.

Residual weaknesses: it does not identify the matched-call versus per-object
opportunity mismatch, gives less weight to prediction-set diversity and merge
asymmetry as primary scientific gates, and spends more space on registry and
process than on the smallest mechanism-discriminating contrasts. Its verdict,
**approve after corrections**, is also premature.

All three reviews are useful. The ranking reflects marginal information gain:
the first changes the causal interpretation, the second changes decode and
aggregation validity, and the third most strongly hardens reproducibility.

## Accepted Findings

The revised unit must incorporate the following findings.

1. **Policy effect, not mechanism localization.** Full-canvas masking changes
   pixels before the vision tower. A positive result establishes masked-input
   spatial-policy utility, not post-vision candidate competition or prior
   recognition in the unmodified full-image state.
2. **Calls, computation, and object opportunities are different budgets.** The
   final policy comparison remains matched by call count, while an owning-seed
   raw object contrast provides the one-opportunity comparison.
3. **Reset versus cumulative is a total prefix-policy intervention.** It
   changes length, content, correctness, visibility consistency, order, and
   state reconstruction. Pure history length requires a later controlled
   prefix panel.
4. **Neutral repetition penalty is primary.** A repetition-penalty value of
   `1.0` is required for the primary mechanism panel. The historical `1.10`
   setting is a separately named sensitivity, not a hidden constant.
5. **Independent bagging requires real sampling and meaningful diversity.**
   Sampling configuration, per-request seed derivation, replay, batching
   invariance, and prediction-set diversity must pass a mechanics gate.
6. **Raw and aggregated outcomes answer different questions.** Owning-call raw
   detection, any-call raw union, post-merge detection, and merge-created or
   merge-destroyed matches must all be retained. The owning-call and final
   policy effects must agree in direction before a scope interpretation is
   strengthened.
7. **Cumulative prompting is a state transition, not string concatenation.**
   Admission, canonicalization, coordinate frame, role, order, invalid rows,
   duplicates, closure, context overflow, and exact prompt bytes and token
   identifiers must be frozen.
8. **Native-tile cumulative history is not a primary arm.** Tile-local
   normalized coordinates make prior rows frame-incoherent. The arm is
   exploratory until an explicit and validated coordinate contract exists.
9. **A mask-harm gate is required.** Equality or a negative masked result cannot
   weaken spatial-scope explanations unless visible core-interior objects are
   retained above a predeclared noninferiority floor.
10. **The reference ledger is sealed before outputs.** The primary
    audit-augmented ledger is produced by arm-blind image review before model
    outputs. Post-output pooled candidate adjudication is a separately named
    secondary ledger and never rewrites the primary denominator.
11. **Official crowd provenance must be restored.** Source `iscrowd` semantics
    are preserved for official evaluation. Crowd regions are not silently
    converted to ordinary individual instances.
12. **Positive rescue requires joint safety.** Retention, manual precision,
    duplicate rate, invalid-row rate, natural closure, and prediction count
    receive predeclared noninferiority or bounded-inflation gates.
13. **One traversal permutation is not a geometry-prior test.** Order analysis
    needs multiple counterbalanced orders and a reset-order mechanics control,
    or it must be described only as sensitivity to one named order.
14. **The rescue baseline seed must be independent.** Reusing a bagging seed in
    the single-rollout miss denominator would condition that comparator to
    failure for one owning cell and bias the raw difference.
15. **Decision states require executable uncertainty rules.** Raw and final
    policy views must each pass their own predeclared minimum meaningful effect
    and superiority rule; same-sign point estimates alone are insufficient.
16. **Terminal output cannot be copied into active state.** The cumulative
    prefix excludes the prior call's terminal no-more-objects and conversation
    termination tokens, and a synthetic test must prove continuation after a
    naturally closed prior call.
17. **Mask construction is numeric protocol.** Color space, arithmetic
    precision, rounding, padding, merged-token support, and the eight-by-eight
    trigger are frozen before execution rather than judged visually afterward.

## Findings Narrowed or Deferred

- Full-image bagging is retained. It is still the correct final-policy utility
  comparator; it is no longer a symmetric per-object falsifier.
- A bagging result equal to a spatial arm means only that no incremental final
  utility was detected under the declared matched-call budget. It does not show
  that spatial restriction is useless.
- Exact superiority, equivalence, precision, and safety thresholds are not
  copied from any review. They must be frozen in the readiness amendment from a
  declared precision target before model outputs are inspected.
- No current review authorizes a ledger, slot, cursor, proposal head, training
  objective, final architecture, or broad OpenSpec change.
- No historical `val200` model score may rank, select, or exclude cohort images.
  It may only size the unit and check that the proposed metric denominators are
  viable.

## Fixed-Point Verdict

**Decision: pass for readiness amendment; not ready for implementation or
execution.**

The first synthesis required revision and re-review. The fixed-point review then
identified independent-baseline-seed bias, an underdefined diversity gate,
non-executable decision terminology, terminal-token leakage into cumulative
state, and underdefined mask construction. The revised unit closes those
design-level blockers and now passes for preparation of the immutable readiness
amendment.

The research unit remains `planned`, with no accepted evidence and no
implementation authorization. The next fixed point was the completed readiness
amendment with exact cohort, ledger-generation, threshold, decode, prompt,
spatial, merge, matching, and budget values, followed by one focused contract
review. Its outcome is recorded below. Only after that outcome may the user
separately authorize a research-only implementation and synthetic mechanics
gate.

The implementation sequence, if later authorized, is:

1. freeze the readiness amendment and sealed reference ledgers;
2. implement synthetic spatial, prompt-state, sampling, matching, and merge
   fixtures;
3. run a tiny non-metric-bearing mechanics gate;
4. run the five-arm primary metric panel;
5. run secondary boundary, order, and repetition-penalty sensitivities only
   under their predeclared triggers.

Every outcome keeps architecture promotion disabled. The strongest positive
masked-input result authorizes a later same-feature post-vision discriminator,
not a final model design.

## Readiness-Amendment Focused Review

### Reviewed artifact

[Readiness Amendment for Masked Spatial Policy and Accepted-Row Prefix Policy
Disentanglement](readiness-amendment.md)

### Independent lanes

The focused review used two independent read-only lanes. Neither lane edited
code, invoked a model, or used a Graphics Processing Unit.

1. The **scientific contract lane** audited causal claim boundaries, cohorts,
   reference ledgers, opportunity matching, decode calibration, seeds, prompt
   state, score, merge, matching, uncertainty, safety, incomplete annotations,
   budgets, and lifecycle.
2. The **executable-semantics lane** compared the desired contract with live
   inference, prompt, image-planning, scoring, merge, parsing, and evaluation
   code using only tiny fake-backend and file/configuration probes.

### Findings closed during fixed-point review

The first scientific pass returned `HOLD` with four high-severity contract
defects. All were closed in the amendment:

1. The accepted-row prefix-policy safety direction now evaluates
   `MASK_RESET` against `MASK_CUMULATIVE`, matching the declared Reset-minus-
   Cumulative estimand, and requires symmetric re-evaluation before claiming a
   safe reverse-direction result.
2. The pre-output Dense-Union-51 review now has a versioned image-only
   instruction packet, exact object states, bounding-box convention, required
   schemas, deterministic assembly, reviewer pairing, adjudication rules, and
   stable object identifiers.
3. Audit-augmented precision now matches accepted individuals first, then
   applies separately counted crowd and uncertainty ignore operations before
   counting unmatched predictions. Official metrics require restored raw crowd
   semantics and a named crowd-aware evaluator.
4. Temperature calibration now uses an explicit two-stage call matrix: up to
   three 48-call candidate panels followed by two 48-call batch-size-four
   replay/order panels for the first tentative candidate. The maximum is 240
   calls and 122,880 configured generated tokens. Batch size four is frozen for
   throughput and batch size one is not a scientific gate.

The review also froze zero-denominator bootstrap behavior, a minimum applicable
replicate count, and an interval-based Prediction-Count Inflation guardrail.
The final delta-only scientific review returned `PASS` with no remaining
critical-, high-, or moderate-severity finding.

### Executed read-only semantics receipts

The executable-semantics lane established:

- the live Hugging Face backend passes `do_sample=false` and does not execute
  configured temperature, nucleus cutoff, or a per-request generator seed;
- the current resolved inference configuration retains temperature `0.0`,
  nucleus cutoff `1.0`, repetition penalty `1.10`, and image-processor
  `do_resize=false`;
- all 200 primary source canvases and all 12 calibration source canvases resolve,
  match declared dimensions, and are divisible by the 32-pixel merged-token
  quantum;
- one live image-materialization fixture preserved its 1,248-by-832 source
  dimensions under `do_resize=false` and produced the expected visual grid;
- the current prompt builder has no accepted-row open-assistant continuation
  surface;
- the repository merge module is a data-parallel artifact-shard merger, not an
  object-level Non-Maximum Suppression implementation;
- the current evaluator rewrites every crowd flag to zero and is therefore
  ineligible for this unit's crowd-aware official metrics;
- the current row score implementation matches Compact Object Selected-Token
  Score, version 1; and
- tile coordinate offset, spatial ownership, masked input, and object-level
  merge remain future research surfaces rather than attested runtime behavior.

This lane returned `PASS` for accuracy of the amendment's blocker inventory and
`HOLD` for execution.

## Final Focused-Review Verdict

**Decision: pass the scientific contract; keep implementation and execution
blocked.**

The readiness amendment contains no unresolved critical- or high-severity
scientific contract issue. It is suitable for a separate user decision about a
bounded implementation/preflight lane. It is not execution-ready because the
sampled request path, open assistant prefix, spatial policies, object-level
merger, crowd restoration, materialized cohort/reference ledgers, hashes, and
run receipts do not yet exist.

The research unit therefore remains `planned`, `evidence_status: none`,
`implementation_status: not_authorized`, and
`architecture_promotion_status: not_promoted`.
