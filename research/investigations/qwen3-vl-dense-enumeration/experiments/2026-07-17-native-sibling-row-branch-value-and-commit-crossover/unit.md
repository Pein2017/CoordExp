---
title: Native Sibling-Row Branch Value and Commit Crossover
description: Natural-prefix pilot testing whether recurrent model-generated sibling rows differ in downstream unique-object value and object-specific commit behavior.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-17-native-sibling-row-branch-value-and-commit-crossover
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: state_local_successor_effect_without_general_commit_or_branch_value_handle
updated: 2026-07-17
---

# Native Sibling-Row Branch Value and Commit Crossover

## Terminology

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model
  family whose native autoregressive detection behavior is under study.
- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed
  set of reportable object categories used by this unit.
- **Weight-Decomposed Low-Rank Adaptation (`DoRA`)**: the adapter mechanism in
  the frozen step-4,887 checkpoint.
- **Gaussian coordinate soft cross-entropy plus Ranked Probability Score**:
  the coordinate objective identified by the historical `gaussian_rps` token
  in the frozen config and artifact paths. The historical token is retained
  only as an identity handle.
- **Historically native parent prefix**: an exact legal assistant continuation
  emitted by the frozen model from the unchanged full-image prompt under the
  source `bf16` runtime. Its complete token identifiers and source request are
  retained. Replaying this fixed prefix under another numerical runtime does
  not establish that the whole prefix is naturally reachable there.
- **Native sibling row**: one complete, valid, naturally closed object row
  sampled from the same native parent prefix. No description, coordinate,
  wrapper, residual state, or visual feature is forced or edited.
- **Physical-entity ledger**: the frozen candidate-conditioned list mapping a
  reportable physical object to one stable identifier, category, and bounding
  box. It is not claimed to exhaust the full scene.
- **Predeclared owner**: any accepted physical-entity ledger owner not already
  covered by the reviewed parent prefix. Discovery-arm admission is not the
  definition of predeclaration.
- **Review-approved absent-ledger entity**: an entity created by a blinded human
  approval with no accepted ledger owner. It is `new_supported_not_predeclared`
  and remains neutral for the first-action `good_event`; it can only contribute
  to the upper unknown-support world.
- **Candidate cluster**: a connected component of same-category unmatched or
  automatically ambiguous
  prediction signatures under the frozen overlap rule. Every member signature
  is retained for deterministic replay matching.
- **Unknown upper bound**: each frozen candidate-cluster identifier counts at
  most once per request, regardless of how many rows or seeds reproduce it.
- **Branch Value at Horizon H (`Q_H`)**: expected number of ledger-supported
  physical entities newly recovered by a sibling row plus at most `H`
  additional naturally generated rows, relative to entities already present
  in the parent prefix.
- **Greedy Branch-Value Gap at Horizon H (`G_H`)**: the largest admitted
  sibling branch value minus the branch value of the greedy sibling at the
  same parent prefix.
- **Commit crossover**: object-specific suppression in which emitting object
  `A` reduces the next-row probability of `A` more than emitting sibling
  object `B`, with the reciprocal result for `B`.
- **Jensen-Shannon divergence**: a symmetric finite divergence between two
  discrete successor distributions. It is used only after subtracting the
  corresponding split-half sampling noise.
- **Prefix State 56 (`P56`)**: the verified image-`12576` natural assistant
  continuation containing 56 generated tokens immediately before the known
  pizza-versus-left-cup branch.
- **Greedy control review row**: an ambiguous or unresolved greedy first row
  inserted into the same blinded candidate packet. Failure to resolve it refuses
  only the Greedy Branch-Value Gap (`G_H`), not sibling crossover.

## Authorization Boundary

The user authorized this complete research unit, bounded implementation,
eight-graphics-processing-unit sampling, adaptive controls, and result
analysis. This authority does not promote model training, a reusable runtime
contract, an architecture, an explicit ledger, an object slot, or a final
decoding policy. Those remain separate decisions.

## Question

At one exact native parent prefix with at least two recurrent native sibling
rows:

> Does the locally preferred greedy row have lower safe downstream
> unique-object value than another naturally supported sibling, and does
> choosing either sibling write an object-specific commit update rather than a
> generic geometry-sorted or row-progression transition?

The branch-value question is primary. Commit crossover is the mechanism-level
secondary question. Cross-row influence horizon is conditional on a positive
native action effect.

The executed evidence and bounded verdict are recorded in
[results.md](results.md).

The formal `G_H` maximum includes the greedy owner itself, whose contrast is
exactly zero, so an identified `G_H` is never negative. Maximum-alternative
contrasts remain separate diagnostics for positive-handle testing, while
`positive_safe_handle` reports whether any alternative passes both safety
worlds. `status: identified` means that the greedy comparison was evaluated; it
does not imply a positive-safe alternative. If greedy evidence is refused,
missing, or incomplete, formal `G_H` fields are null and the zero baseline is
marked not applicable; zero is reserved for a valid comparison in which greedy
is the maximizing admitted owner.

## Why This Unit Exists

The preceding hard-coordinate transport unit stopped because the compared
intermediate histories did not share model support. Coherent off-support output
was not causal identification. This unit changes the treatment from an edited
coordinate or hidden state to a complete row that the model itself recurrently
generated from the exact same parent state.

Earlier evidence establishes the starting handle but not the answer:

- at image `12576` `P56`, 32 samples split into target pizza `9/32` and left
  cup `23/32`, while greedy selected the left cup;
- the pizza and an independent chair disappeared at later natural terminal
  prefixes;
- teacher-forced complete rows produced a strong successor transition but
  failed strict object-specific commit-to-uncovered redistribution; and
- the 119-candidate manual audit found abundant semantic support but frequent
  fragmented, multi-owner, and axis-mixed geometry, so physical ownership must
  gate every commit claim.

## Competing Explanations and Predictions

### Local likelihood versus downstream set value

The decoder chooses the most likely immediate row, while the chosen branch may
have lower expected future unique-object coverage. A lower-frequency admitted
sibling should then have larger safe `Q_H` than the greedy sibling. The
explanation is falsified if greedy is consistently branch-value optimal or all
admitted siblings have indistinguishable downstream value.

### Native object-specific commit

A complete native row writes the emitted physical entity into executable state
that selectively suppresses that entity and redistributes probability to
other uncovered entities. Both directions of the sibling-object commit
crossover should be positive. Equal successor distributions, non-reciprocal
suppression, or mass moving mainly to terminal, invalid, or duplicate outcomes
falsifies strict object-specific commit.

### Geometry-sorted serialization or traversal frontier

The prefix records progress along the training order rather than an order-free
covered set. After a later-ranked sibling is emitted first, the model should
continue forward instead of returning to an earlier omitted sibling. Several
different rows may share the same canonical successor.

### Generic legal-row progression or concentration

Any valid row reduces successor entropy or advances toward one dominant mode,
without physical-entity-specific state. Different admitted siblings should
produce similar concentration, terminal, or successor movement.

### Fragile prefix order or list inertia

Earlier row order remains active beyond the set of entities already emitted.
Naturally supported histories with the same covered-entity set, same row count,
and eventually the same last bridge row may retain successor distributions
above split-half sampling noise.

### Simple independent-sampling support

Bagging improves only because it purchases more independent draws. Equal-call
suffix unions should then be similar across sibling actions and `G_H` should
be negligible.

### Annotation, owner, or numerical ambiguity

Apparent rescue may be a missing label, fragment, category disagreement,
multi-owner box, reduced-precision branch, or physical-batch artifact. Every
commit claim therefore requires a frozen single-owner entity, physical-batch-
one execution, and a 32-bit floating-point robustness check for a conclusion-
changing effect.

## Frozen Model and Decode Surface

Primary config:

```text
configs/coordexp_infras/infer/
qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml
```

The source-consistent behavioral lineage is:

- description-first, geometry-sorted compact object rows;
- step-4,887 Gaussian coordinate soft cross-entropy plus Ranked Probability
  Score DoRA adapter;
- unchanged full image with resizing disabled;
- Brain Floating Point 16-bit model parameters;
- Scaled Dot-Product Attention;
- physical batch size one;
- sampling temperature `0.4`;
- top-p nucleus threshold `0.95`;
- repetition penalty `1.0`;
- greedy control with sampling disabled and repetition penalty `1.0`.

The physical batch size is fixed at one because an earlier unit proved that a
Brain Floating Point 16-bit batch-shape change can change an object basin.
Selected scores and reductions use 32-bit floating point. A separately loaded
full-model 32-bit floating-point, batch-one lineage is a numerical robustness
control for conclusion-changing effects; it does not make `P56` an end-to-end
native 32-bit trajectory. Eight graphics processing units parallelize
independent batch-one seed shards; they do not form one physical batch.

## Frozen Cohort

| Image identifier | Role |
|---|---|
| `12576` | Known `P56` pizza-versus-left-cup fragmentation and first smoke. |
| `2299` | Dense repeated-person scene with 49 accepted ledger entities. |
| `9400` | Laptops, people, keyboards, and category-versus-geometry disagreement. |
| `7574` | Same-category bowl, bottle, glass, and cup competition. |

Conditional controls run only after the first primary state passes:

| Image identifier | Role |
|---|---|
| `19432` | Dense-chair stress; excluded from headline ownership unless a selected row has one unambiguous ledger owner. |
| `15254` | Prior no-unmatched-candidate control for transition and terminal behavior. |

The accepted augmented ledger is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-13-spatial-scope-history-disentanglement/readiness-v2/
audit-augmented-ledger.jsonl
```

Only entities touched by admitted branches receive additional targeted visual
review. Before owner-level branch comparisons are unblinded, every unmatched
or automatically ambiguous COCO-80 suffix candidate from every admitted arm is adjudicated with
branch and seed hidden. The review freezes stable entity identifiers and each
parent's covered set while preserving an explicit unknown outcome. The unit
does not reopen all 119 historical predictions and does not claim exhaustive
scene recall.

An ambiguous automatic parent-row match may be resolved by its frozen review
candidate; otherwise the parent remains unresolved and owner-level completeness
is refused. The same rule applies to an unresolved greedy first row, which is
added to the blind packet when available. Greedy failure alone refuses only
`G_H`, not sibling crossover.

If review maps a row to an accepted ledger owner, accepted-ledger membership
overrides review provenance and the row is predeclared. Only a review-approved
entity absent from the accepted ledger is not-predeclared. For branch-value
counts, not-predeclared entities contribute zero to the lower world and one to
the upper world, just like a frozen unknown candidate.

## Frozen Randomization

- discovery root seed: `2026071701000001`;
- confirmation root seed: `2026071702000001`;
- paired-bootstrap root seed: `2026071703000001`.

The experiment-local seed schedule is a deterministic hash of root seed,
image identifier, exact parent-prompt hash, and within-panel seed index. It is
independent of sibling or row-variant labels, so all arms at one parent receive
the same confirmation seeds. Discovery and confirmation roots never mix. The
manual seed `2026071701` used by the initial one-call runtime smoke is excluded
from all scientific estimates.

## Wave Zero: Artifact-Only Native Trajectory Atlas

Before new model calls:

1. combine the two existing 16-call Full-Image K-Rollout Independent Bagging
   seed roots into 32 trajectories per selected image;
2. map valid rows to accepted ledger entities, unresolved real support,
   duplicate or fragment, unsupported output, invalid row, or terminal;
3. record each accepted entity's inclusion frequency, first-hit row, immediate
   predecessor, and downstream co-occurrence;
4. identify legal natural parent prefixes with at least two candidate sibling
   actions; and
5. freeze discovery and confirmation seed roots before new sampling.

Replay uses exact prediction-signature membership from the frozen candidate
clusters. Connected-component bridge cases remain one candidate even when the
first and last boxes do not overlap directly.

Existing artifacts select candidate states; fresh sampling owns every new
mechanism claim.

## Wave One: Exact-Prefix Native Sibling Discovery

The first smoke is image `12576` `P56`, treated as a fixed historically native
parent prefix under the source-consistent runtime.

1. Rebuild the exact natural parent prefix from its source terminal bundle and
   verify prompt-plus-continuation token identity.
2. Run greedy plus 32 independent one-row samples at physical batch size one.
3. If a plausible candidate appears only once or twice, expand the state to 64
   discovery samples without changing the support rule post hoc.
4. Preserve every child row exactly as emitted, including phrase, coordinates,
   wrappers, source request, and seed.
5. Freeze recurrent sibling action groups by physical owner before suffix
   replay.

A sibling action is support admitted only when:

- it was naturally generated from the exact parent prefix hash;
- its row is valid and closed naturally;
- it maps to one frozen physical entity;
- its owner group occurs at least `3/32`, or `4/64` after expansion; and
- at least three distinct naturally generated row instances support the owner
  group.

No edited row, synthetic coordinate, ground-truth continuation, cross-prefix
row, residual replacement, visual mask, or constrained token is admissible.

## Wave Two: Native Sibling Branch Value

For each parent with at least two admitted sibling owners `A` and `B`:

1. retain every distinct naturally emitted exact row variant per admitted
   owner, together with its frozen discovery count;
2. rebuild each exact `parent plus child` prefix without editing the child;
3. run eight fresh confirmation seeds per row variant;
4. decode at most four additional complete rows or natural termination; and
5. classify every row against the frozen ledger and targeted visual review.

For parent `s`, one exact naturally emitted sibling-row variant `r`, and
horizon `H=4`:

```text
q_H(s, r) = expected count of newly recovered unique supported entities
            in r plus at most four subsequent complete rows

Q_H(s, o) = sum over exact row variants r owned by o of
            discovery_frequency_weight(r | o) times q_H(s, r)

G_H(s) = maximum admitted-owner Q_H(s, o)
         minus Q_H(s, owner of the fresh greedy row)
```

Exact-row `q_H` is always reported. Every discovered exact variant of an
admitted owner is executed, so owner-level `Q_H` uses the full empirical owner-
conditional distribution with weights frozen from discovery counts before
confirmation; equal row-variant weights are a sensitivity analysis only. If
any discovered variant is missing, fails execution, or lacks its eight
confirmation seeds, owner-level `Q_H` and `G_H` are refused while completed
exact-row `q_H` values remain descriptive. The greedy owner is eligible for
`G_H` only when the fresh source-consistent batch-one greedy row is valid,
support admitted, single owner, and uncovered at the parent. Otherwise `G_H`
is not identified for that state.

Report separately the difference in unique supported entities and the
differences in unsupported, invalid, duplicate-current, duplicate-covered,
unknown, and terminal outcomes. Confirmation seeds alone own these estimates.
For the `0.10` safety gates, each outcome rate is the per-confirmation-request
probability that the outcome occurs at least once within the four-row horizon;
it is not normalized by the number of generated rows. Row-level counts remain
descriptive only, so a branch that terminates early cannot appear safer merely
because it exposed fewer row opportunities.
For each owner contrast, use two-sided 95% paired-bootstrap intervals. The
bootstrap unit is one confirmation-seed index, resampled jointly across all
row variants and sibling-owner arms; frozen discovery-frequency weights are
reapplied inside each replicate. Use a simultaneous max-statistic interval
across admitted non-greedy owners. A positive branch-value handle requires the
lower 95% confidence bound of the unique-entity difference to exceed zero,
while upper 95% confidence bounds for each of unsupported, invalid, and total-
duplicate rate differences do not exceed `0.10`. Duplicate-current,
duplicate-covered, unknown, and terminal contrasts are reported descriptively
and are not silently folded into the safety gate. Apply the same tests to both
unknown-outcome lower and upper estimands. No scalar reward combines these
outcomes.

Unknown adjudications produce two branch-value bounds: a lower bound treating
them as unsupported and an upper bound treating them as new supported
entities. A positive handle is refused when the sign of the owner contrast
changes between these bounds.

## Wave Three: Commit Crossover

For every unordered pair of admitted sibling owners `A` and `B`:

```text
C_A = probability(next owner is A | parent plus B)
      minus probability(next owner is A | parent plus A)

C_B = probability(next owner is B | parent plus A)
      minus probability(next owner is B | parent plus B)
```

Object-specific commit requires both crossovers to be positive in the same
state, released mass to favor predeclared uncovered entities rather than bad
outcomes, replication across at least three parent states and two images, and
no reversal across same-owner row variants. Every unordered owner pair is
reported; a pair is not silently replaced by an aggregate multi-owner
contrast. Every probability uses all
requests, including terminal, invalid, unsupported, duplicate, and unknown
outcomes in the denominator. The same fresh confirmation seed set is paired
across sibling owners and row variants. Report every exact-variant pair
crossover with paired-bootstrap intervals for both `C_A` and `C_B`; both lower
bounds must exceed zero, and no exact-variant reversal is allowed, to support a
state-level crossover. No-reversal is assessed from the paired interval for
each exact row-variant pair, not by requiring every individual seed contrast to
be nonnegative. One failed state rejects only that state. Mixed panel evidence is inconclusive; panel support requires the
declared three states across two images, while failure in all admitted states
closes strict native object-specific commit for this panel.

For each committed owner `A`, define `good_A` as the unconditional probability
that the next action is a different predeclared uncovered parent-ledger entity.
Define `bad_A` as the unconditional probability of terminal, invalid,
unsupported, previously covered, repeated-`A`, or unknown outcomes. Strict
commit-to-uncovered redistribution requires both crossover lower 95% bounds to
exceed zero and, for both owner arms, the lower 95% bound of
`good_A - bad_A` to exceed `-0.10`. Object-specific suppression may still be
reported separately when this redistribution noninferiority gate fails.

## Conditional Wave Four: Natural Influence Horizon

Enter only if Wave Two establishes a reproducible sibling-action effect. Search
for a naturally supported order diamond:

```text
parent -> A -> B
parent -> B -> A
```

Every edge must pass the native support gate. If possible, append one common
native bridge row `C` recurrent after both histories. The resulting prefixes
then have the same covered set, row count, and last row but different earlier
order. Compare distance-one-through-three successor distributions using
Jensen-Shannon divergence minus split-half noise. Do not fabricate a bridge
when no common native support exists.

## Implementation Outline

Reuse only the prompt, assistant-continuation, parsing, seed-schedule patterns,
and compact-bundle helpers from:

- `src/analysis/sampled_rescue_transition/` for terminal bundles, chronology,
  and row spans;
- `scripts/research/run_sampled_rescue_transition.py`;
- `src/inference/backend.py` for sampled generation and receipts;
- the accepted augmented ledger and existing trajectory artifacts; and
- current parsing and visualization helpers.

Add only experiment-local surfaces to build the parent/sibling atlas, execute
source-consistent and full-model-32-bit batch-one seed shards, merge collision-
safe shards, and emit exact-row plus owner-level branch-value and crossover
tables. The experiment-local runner must not route sampled requests through
the old batch-four execution loop. The source-consistent lineage may rebind the
existing policy/runtime capability, but every actual scientific call must
contain exactly one request; a real one-request smoke receipt, rather than the
old attestation cases alone, owns this cardinality claim. Its first receipt
must attest actual model parameter dtypes, Scaled Dot-Product Attention, every
physical request cardinality of one, exact executed decode arguments, request-
scoped seeds, and request identifiers. A two-shard replay must prove seed
collision freedom and stable request attribution before scientific sampling.
Do not change the stable inference schema or public pipeline.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-17-native-sibling-row-branch-value-and-commit-crossover/
<immutable-run-identifier>/
```

Each receipt records model and prompt identities, image digest and dimensions,
exact prefix hash, source trajectory and row span, dtype, attention
implementation, physical batch size, seed, decode arguments, parser status,
raw rows, ledger digest, discovery-versus-confirmation split, and terminal
status.

## First Real Smoke and Rough Cost

Image `12576` `P56`: one source-consistent Brain Floating Point 16-bit process,
one graphics processing unit, physical batch size one, one greedy request, and
32 one-row samples. If both pizza and left-cup owners pass support, run their
paired four-row suffix confirmation. A conclusion-changing effect then receives
one full-model 32-bit floating-point, batch-one robustness replay; this replay
is conditional on the fixed prefix and does not claim end-to-end native 32-bit
reachability. Only then distribute admitted states across up to eight
independent graphics-processing-unit workers. The full primary panel remains
below 256 parent-prefix discovery samples plus admitted suffix calls.

## Stop Rules

- Stop before suffix replay if `P56` lacks two support-admitted native sibling
  owners under the primary runtime.
- Stop the unit if fewer than three primary-cohort parent states contain two
  admitted sibling actions.
- Do not propose branch-value training when no safety-preserving `G_H` appears
  after three admitted states.
- Close strict native object-specific commit when reciprocal crossover fails
  across all admitted states.
- Report influence horizon as unidentifiable when no natural order diamond or
  common bridge exists.
- Exclude owner-ambiguous, fragmented, or multi-entity states rather than
  weakening the ledger.
- Classify an effect as runtime- or precision-sensitive if it changes under
  the conditional batch-one 32-bit floating-point robustness replay; do not
  generalize it as a stable native mechanism.
- Do not enter hidden-state analysis until a native sibling action has a
  reproducible downstream effect.
- Do not start training, architecture work, or a larger cohort from a
  descriptive transition alone.

## Non-Goals

- no population mean Average Precision, recall, hallucination, or prevalence;
- no claim that the ledger exhausts every visible object;
- no claim that candidate competition is specifically visual;
- no claim that sampling is the final detector policy;
- no repetition-penalty, terminal-suppression, or temperature optimization;
- no hidden-state, attention, residual, or object-slot intervention;
- no conclusion that an explicit covered-set carrier is necessary; and
- no training or architecture promotion.

## Execution Closeout

The unit completed discovery and confirmation on four admitted states. Three
states released owner-level conclusions and one dense-person state was
correctly refused because a parent-prefix row remained physically unresolved.
No non-greedy sibling produced a positive, safety-preserving Greedy
Branch-Value Gap at Horizon Four (`G_H`) after the required three released
states, so the branch-value training gate is closed.

One row-zero bowl-versus-carrot state on image `15254` passed strict reciprocal
commit crossover in source-consistent Brain Floating Point 16-bit execution.
The same 32 exact sampled rows and same eight paired confirmation seeds were
then replayed with all model parameters in 32-bit floating point; the effect
reproduced with no exact-variant reversal. This is a verified state-local
successor effect. It is not promoted as a general physical-object covered set
because two other released states reject reciprocity, the dense-person state
is refused, and the positive bowl and carrot owners occupy nearly identical
spatial support.

Conditional influence-horizon work does not run inside this unit. The next
high-information discriminator, if separately authorized, is a three-owner,
same-category, spatially non-overlapping native transition matrix with fully
resolved physical ownership. Training, architecture, hidden-state analysis,
terminal suppression, and repetition-penalty optimization remain outside this
closeout.
