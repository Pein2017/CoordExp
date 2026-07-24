---
title: Frozen-Panel Image-First Owner-Ledger Adjudication Salvage Gate
description: A train-only category-capacity upper screen followed, only if necessary, by a blinded probability sample testing whether image-first owner-ledger adjudication can recover the 248 additional admissions required by the frozen 17-candidate panel.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: in_progress
unit_id: 2026-07-23-trajectory-owner-set-adjudication-salvage-gate
topic: qwen3-vl-dense-enumeration
status: active
evidence_status: frozen_contract_approved
updated: 2026-07-23
---

# Frozen-Panel Image-First Owner-Ledger Adjudication Salvage Gate

## Decision Boundary

The preceding root-state census is closed. It found 8 primary-admitted images
among 2,004 Source-eligible training images, versus 256 required, and therefore
rejected training promotion. Four admissions are fully adjudicable and four
are censored. The 1,622 remaining censored, nonpassing images would need to
contribute at least 248 new admissions for the original cohort target to become
possible.

This successor was selected only after that result was complete. Independent
science, infrastructure, quantitative, and adversarial lanes agreed that the
highest-value unresolved question is attribution coverage under the unchanged
trajectory panel. More trajectory exploration would change support while the
attribution boundary remains unknown. Architecture or objective work would
change still more causal surfaces and is therefore downstream.

This unit asks whether a complete, route-blind, image-first physical-owner
ledger followed by the existing global matcher can recover the required 248
additional admissions. It does not reopen the original admission predicate,
select training data, or authorize training.

## Falsifiable Question and Estimand

Let `U` be the exact 1,622 training images that are both censored and not
primary-admitted in the completed census. Under the canonical adjudication
protocol frozen below, are at least 248 members of `U` actual or conservatively
unresolved potential admissions under the unchanged 17-candidate panel?

The null used for a futility rejection is:

```text
H0: at least 248 images are protocol successes.
```

A protocol success is either:

1. an actual primary admission after the sealed owner ledger is globally
   replayed; or
2. a potential admission whose image-first adjudication remains unresolved in
   a way that could change ledger membership, row assignment, owner identity,
   first owner, trusted geometry, a safety count, class or edge construction,
   frontier status, or final admission.

Unresolved cases count as successes, never failures. This makes a futility
rejection conservative with respect to review uncertainty. The estimand is the
outcome of this frozen protocol, not the number of existentially imaginable
owner assignments.

Failure to reject is inconclusive. It does not estimate a 248-image cohort,
authorize broader adjudication, or authorize training.

## Frozen Inputs and Population

The execution must bind these exact completed-census artifacts:

| Input | Frozen identity |
| --- | --- |
| Image census | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-admission-census/production-v1/image-census.jsonl`; SHA-256 `c375e09df621714da82118a2cdf3d80205fd7429424b1a853f38ec60e68e1805` |
| Census summary | same root, `summary.json`; SHA-256 `2cc526310e8bca2de11f2d9d7c2e42511a017a882455dd1ebdb671b0c0df4514` |
| Census receipt | same root, `receipt.json`; SHA-256 `d322335c5e92f3fd4860b9524b8d709a3d01fe9e1cbc625dfa1d4bf002d12fe2` |
| Candidate pool | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/candidate-pool-v1/candidate-pool-2432.coord.jsonl`; SHA-256 `133afcf6659b78d71893e9f79e568dca676515a05c7fe23f3c237de01350b7e2` |
| Sampled trajectories | exact `trajectory-panel-2432-vllm/production-v2` root and manifest-set SHA-256 `28304194b0b5e2aa7a718b2ee080253f616a8d9cf8fde8653e335589b3179b60` |
| Source trajectories | exact `source-b16-vllm/production-v1` root and manifest-set SHA-256 `4e979f920798dcd338c0620500d36b007f90404bb4a2852014702de93b9f521e` |

`U` is derived mechanically from the image census and must contain exactly
1,622 unique identifiers satisfying both:

```text
fully_adjudicable == false
admission.primary_natural_alias_admitted == false
```

The analyzer must reproduce all original split, inventory, model, tokenizer,
projection, and source-snapshot checks. Development and held-out identifiers
are filtered before route-semantic inspection. No development or held-out
route record or summary may be produced.

## Stage Zero: Category-Capacity Over-Approximation

Stage Zero removes images only when immutable parser, token, and generated-row
category facts make primary admission impossible even under an intentionally
optimistic physical-owner assignment. It must not use current owner
identifiers, current owner sets, current eligibility, current edges, current
first-owner identities, official annotation count, geometry, safety, frontier
status, or density to exclude an image.

For each candidate `c`, retain only:

- projected generated-token SHA-256 `tau_c`;
- exact `B16` parser usability;
- the ordered normalized category sequence of its evaluated complete rows.

A candidate is parser-usable only when its exact `B16` projection status is
`accepted_budget` or `accepted_natural_end`, its parser status is `accepted` or
`accepted_with_drops`, and no malformed or dropped material occurs before the
frozen projection boundary. Malformed or dropped material strictly after the
boundary does not disqualify it. Owner uncertainty, ambiguity, and geometry do
not disqualify it in this relaxation.

Collapse exact projected-token duplicates. Every duplicate must have the same
ordered category sequence and parser usability or execution fails.

For candidate `c`, define category support `S_c`, category multiplicity
`n_c[g]`, and first-row category `f_c`. A nonempty candidate may optimistically
realize any owner set with exactly support `S_c` and between one and `n_c[g]`
distinct owners of every category `g`. Multiple rows may optimistically map to
one owner. Owner identities, boxes, image-level coupling, and assignment costs
are otherwise relaxed.

An image enters the possible pool if exhaustive search finds parser-usable,
distinct-token representatives `a`, `b`, and `k` such that `a` and `b` can be
aliases of one higher owner set and `k` can realize a strict lower subset.
For a nonempty lower candidate, a higher alias `d` must also preserve its first
owner. `d` may equal `a` or `b`; it need only be token-distinct from `k`.

The exact conditions are:

```text
S_a == S_b
tau_a != tau_b
tau_k not in {tau_a, tau_b}
S_k is a subset of S_a
```

For nonempty `S_k`, additionally require:

```text
S_d == S_a
tau_d != tau_k
f_d == f_k
```

Let `D = {a, b}` when `S_k` is empty and `D = {a, b, d}` otherwise, and let:

```text
u[g] = min(n_x[g] for x in D)
```

Higher first-owner diversity requires either `f_a != f_b` or
`u[f_a] >= 2`. Strictness requires either `S_k` to be a proper subset of
`S_a`, or at least one category with `u[g] >= 2`.

When `S_k` is empty, the lower owner set is empty, strictness follows from the
nonempty higher first owners, and lower-first preservation is vacuous. No `d`
witness is required. This branch is part of the general executable contract
even if the frozen population contains no parser-usable zero-row candidate.

These conditions are necessary, not sufficient. Same-category matching means
every owner in a realized higher set consumes at least one row of that category
in every higher alias. Different physical first owners require either different
first categories or two same-category owners. A strict lower set has category
support contained in the higher support, and equal support requires higher
multiplicity somewhere. Lower-first preservation supplies `d` for a nonempty
lower set. Geometry, physical feasibility, global matching coupling, universal
safety, all other aliases, and nondomination only remove realizations, so
relaxing them may create false positives but may not create false negatives.

For every member of `U`, Stage Zero must emit either:

- one canonical witness tuple proving category-capacity possibility; or
- a machine-replayable exhaustive-search certificate showing that no valid
  tuple exists.

The production artifact must report category-state, possible-pool, and
certificate hashes; forward and reversed input traversal must produce
identical semantic hashes. The production result is not accepted until an
independent reviewer replays every excluded-image certificate against the
frozen inputs.

If the possible-pool size `N` is less than 248, the frozen panel is
deterministically unable to supply the missing cohort and the unit stops. If
`N` is at least 248, only members of this fixed over-approximation may enter
the probability sample.

### Reconnaissance disclosure

Before this unit was frozen, two read-only implementations explored several
candidate relaxations. Owner-set-based pools of 576 and 606 were withdrawn
because global re-matching can remove a previously assigned owner. A later
category-only scout produced 1,106 possible images, but this count and its
hashes are not scientific evidence. The production Stage Zero must reconstruct
the result from the frozen contract and survive independent replay. The
contract may not be changed to recover a preferred pool size.

## One-Time Probability Sample

Only after the Stage Zero directory is complete, immutable, and independently
approved may selection begin. Before entropy acquisition, the image-only
reviewer packet, ontology and state definitions, reviewer and adjudication schemas,
adjudication rules, replay implementation, outcome classifier, and all their
explicit and transitive source or artifact hashes must also be frozen and
recorded. This frozen-source manifest must include the selector implementation
itself and every source that can change review, adjudication, replay, or outcome
semantics. The selector must bind the complete independently audited Stage Zero
directory: exact root inventory, every output hash and size, category-state and
possibility-census artifacts, replayable witness or impossibility certificates,
both pools, summary, source snapshot, counts, and independent replay evidence.

Before the journal is claimed, the selector must also validate and bind one
canonical manifest over every numerically ordered possible-pool member. Each
entry contains the image identifier, canonical source-canvas path, byte size,
SHA-256, width, height, and the complete official-owner and replay-input record
hash. No per-image input may be replaced, admitted, or first validated after
entropy persistence. Uniformity is conditional only on `R < L`; any later
per-image input mismatch voids the unit without a redraw.

The unit has exactly one canonical journal path. Before entropy acquisition,
the selector must exclusive-create a claim containing the unit hash,
frozen-source-manifest hash, complete Stage Zero inventory hash, possible-pool
member-manifest hash, `N`, ordered-pool hash, selector hash, and canonical
selection output root. It must fsync the claim, fsync its parent directory, and
read the claim back successfully. Once this journal exists, no process may
acquire entropy again.

Only then may the selector read one 64-byte operating-system cryptographic
entropy block exactly once. It must durably persist and read back exactly that
block before semantic sample materialization. After interruption, selection
resumes only when the same journal contains exactly 64 readback-valid entropy
bytes. Otherwise the terminal status is `entropy_persistence_unknown`,
including any operating-system or filesystem failure before that proof; the
unit is void and the entropy block must not be redrawn.

Any semantics-changing correction to the review, adjudication, replay, or
outcome-classification contract after entropy persistence voids this inferential
unit. It may not be repaired by keeping a favorable subset, changing a cutoff,
or drawing replacement entropy. A selected-image smoke may validate only the
already-frozen semantics. A semantic, source-manifest, or Stage Zero audit
mismatch after entropy persistence likewise voids the unit and forbids a redraw;
a later mechanical failure resumes the same journal only when the exact entropy
block and every pre-entropy binding remain readback-valid.

Let `N` be the realized possible-pool size and let

```text
M = P(N, 32) = N * (N - 1) * ... * (N - 31)
R = the unsigned big-endian integer encoded by the 64 entropy bytes
L = floor(2^512 / M) * M
```

If `R >= L`, publish terminal status `entropy_rejected`, select no image, and
stop permanently without a redraw. Otherwise set `rank = R mod M` and unrank
one ordered 32-image sample lexicographically from the numerically ordered
possible pool. At ordered position `i` from zero through 31, compute
`suffix_count = P(N - i - 1, 31 - i)`, choose zero-based remaining-item index
`rank // suffix_count`, remove that item from the remaining pool, and replace
`rank` by `rank mod suffix_count`. Conditional on selection publication, every
ordered 32-image sample therefore has exactly equal probability.

The first 16 images are Look One. Images 17 through 32 are the additional
Look-Two block and are reviewed only if Look One does not reject. The artifact
schema must expose `look_one_image_ids`, `look_two_additional_image_ids`, and
`look_two_cumulative_image_ids`; the last is exactly the ordered first 32 and
is the inferential Look Two.

For the realized `N`, compute the largest integer cutoff whose exact
hypergeometric lower-tail probability under `K=248` is at most 0.025 at each
cumulative look. The formula and implementation are frozen before entropy
acquisition. Read-only reconnaissance predicts these cutoffs for
`N=1,106`:

| Cumulative look | Futility rejection | Boundary-null probability |
| --- | ---: | ---: |
| 16 | zero protocol successes | 0.0166704461 |
| 32 | at most two protocol successes | 0.0142553673 |

Look Two uses the cumulative first 32, not only the second block. Bonferroni
controls family-wise Type I error below 0.05. The selector and statistics
receipt must record `N`, `K`, sample sizes, cutoffs, boundary probabilities,
the entropy bytes, `M`, `R`, `L`, the acceptance or terminal-rejection result,
the exact unranking protocol, accepted initial rank, unranking trace, ordered-
pool hash, ordered-sample hash, the three selected-identifier views, the
complete frozen-source manifest, and
the complete Stage Zero audit binding. A completed selection directory may
become visible only after terminal validation; any invalid post-rename root
must be quarantined before a validated failure receipt is published.

## Image-First Review and Adjudication

Two independent reviewer roles inspect every selected image. Before their role
artifact is sealed, each reviewer may receive only:

1. the complete unmodified source-canvas image;
2. image identifier, byte hash, width, and height;
3. assigned reviewer role and review identifier;
4. the frozen Common Objects in Context 80-category ontology; and
5. the image-only reviewer instruction packet and its hash.

Reviewers must not receive official boxes or counts, model predictions,
candidate or route identifiers, decode modes, token data, census labels,
category-capacity witnesses, sample cutoffs, cumulative outcomes, crops,
overlays, or the other reviewer's work. Each role exhaustively enumerates
visually separable Common Objects in Context instances using source-canvas
pixel-edge boxes and the existing accepted, ambiguous, partial, crowd, and
out-of-scope states. Every selected image requires an explicit disposition.

Only after both role artifacts are sealed may a separate adjudicator inspect
both labels and official annotations. The adjudicator may receive only the
source canvas, both sealed role artifacts, frozen official-owner records,
ontology, and frozen adjudication packet. It must not receive candidate, route,
Stage Zero, sample-order, look-membership, cutoff, prior-outcome,
cumulative-count, or sequential-decision information. Adjudication for the
current look must be sealed before replay or outcome classification begins.
The adjudicator identifies reviewer proposals already owned by official
individuals, reconciles duplicate proposals, retains unresolved ambiguity,
and adds only new physical owners. Official owner identities and boxes are
immutable. Adjudication may not edit a model prediction, category, token
sequence, parser result, current candidate, owner set, edge, or outcome.

The sealed audit-augmented owner ledger contains official individuals plus
accepted missing physical owners with one normalized category and one
source-canvas box each. Ambiguous, partial, crowd, unresolved-disagreement, and
ledger-completeness-uncertain records remain explicit uncertainty. An
unmatched official annotation is never automatically a hallucination, and a
reviewer proposal duplicating an official object is not a new owner.

## Frozen Global Replay and Outcome Labels

After the image-first ledger is sealed, load only the selected train image
identifiers through the existing exact `B16` adapter. Merge accepted additions
with official owners and run the existing category-constrained global
`match_prefix` assignment for all 17 frozen candidates. Do not overlay owner
identifiers onto existing candidate records and do not preserve previous
assignments by fiat.

Recreate candidate receipts and image admission records through the original
`_candidate_receipt` and `_analyze_image_candidates` semantics:

- entity owner intersection over union at least 0.50;
- trusted geometry intersection over union at least 0.75 for every row;
- exact projected-token deduplication;
- whole-candidate parser, ambiguity, unknown, harm, and geometry exclusion;
- universal alias safety;
- lower-first-owner preservation;
- strict owner-set inclusion; and
- same-class, multiple-first-owner, nondominated primary admission.

Every replay must preserve candidate identifiers, route count, projected and
raw token hashes, decode identity, row order, and parser evidence. A no-op
ledger must reproduce the original census record exactly.

Assign exactly one image outcome:

- `actual_admission`: primary admission is true after replay;
- `potential_admission_unresolved`: primary admission is not established, but
  sealed review uncertainty could change ledger membership, row assignment,
  owner identity, first owner, trusted geometry, any safety count, class or
  edge construction, frontier status, or final admission; or
- `definitive_non_admission`: the ledger is complete, replay is valid, primary
  admission is false, and no retained review uncertainty can change any owner-
  universe, assignment, safety, class, edge, frontier, or admission fact.

The hypergeometric success count is the sum of `actual_admission` and
`potential_admission_unresolved`. Review disagreement, missing disposition,
invalid geometry, artifact drift, or adjudication failure may never be
converted into `definitive_non_admission`.

## Stop and Interpretation Rules

1. If Stage Zero yields `N < 248`, stop with deterministic frozen-panel
   impossibility.
2. Otherwise execute Look One. If its conservative success count is at or
   below the frozen cutoff, reject `H0` and stop.
3. If Look One does not reject, complete the second 16 images. If the
   cumulative conservative success count is at or below its cutoff, reject
   `H0`; otherwise stop as not falsified.
4. Any rejection means only that this frozen 17-candidate panel plus this
   frozen image-first protocol cannot supply 248 additional admissions at
   family-wise alpha at most 0.05.
5. Nonrejection is inconclusive and authorizes neither a larger review nor
   training without another explicit unit.

No branch may relax the admission predicate, count synthetic row
permutations as natural aliases, recollect trajectories, inspect development
or held-out route semantics, run model inference, use a graphics-processing
unit, train a model, choose a loss, or choose an architecture.

## Implementation, Smoke, and Publication Gates

The smallest implementation is one experiment-local Stage Zero and selection
analyzer plus one experiment-local review/adjudication/replay assembler. Reuse
the exact panel adapter, global matcher, census candidate/image analyzers, and
the existing image-only ledger validation patterns. Do not generalize stable
review infrastructure merely for this pilot.

Required pre-production checks include:

- focused unit tests for parser boundary, exact-token collapse, category
  multiplicity, empty lower support, `d == a`, strictness, lower-first
  preservation, certificate replay, randomization, hypergeometric cutoffs,
  review blindness, unresolved-as-success, and no-op global replay;
- static checks and source diagnostics on every changed Python file;
- a three-image Stage Zero fixture with one possible image, one certified
  impossible image, and one parser-invalid candidate;
- no-op replay of censored admission image `2434`;
- geometry exclusion retention for image `831`; and
- one genuinely reviewed selected image before a complete look is assembled.

Publication roots are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-23-trajectory-owner-set-adjudication-salvage-gate/
  stage-zero-v1/
  selection-v1/
  look-one-review-v1/
  look-two-review-v1/  # only when Look One does not reject
```

Each root is directory-atomic, has a terminal receipt, exact input and output
hashes, task-scoped source snapshots, readback validation, deterministic
semantic hashes where applicable, immutable success permissions, and no
sibling staging residue. Selection must additionally prove one-entropy-draw
resume semantics. Before use, every review or replay assembler must revalidate
the canonical journal, terminal selection success receipt, complete selection
root inventory, and all hashes. The Look-One queue must equal ordered positions
1 through 16; the Look-Two additional queue must equal positions 17 through
32; and the cumulative Look-Two decision must bind the accepted immutable
Look-One root plus ordered positions 1 through 32 exactly. A post-publication
mismatch blocks inference. Mechanical reconstruction may use only the same
sample and sealed inputs, never replacement images, redrawn entropy, or a
favorable subset. Review roots must bind the packet, ontology, queue, both role
artifacts, adjudication, owner ledger, replay records, outcome labels, and exact
sequential decision.

An independent advanced-model review must approve the frozen contract before
implementation, the Stage Zero artifact before random selection, and any
futility rejection before it becomes a scientific conclusion.
