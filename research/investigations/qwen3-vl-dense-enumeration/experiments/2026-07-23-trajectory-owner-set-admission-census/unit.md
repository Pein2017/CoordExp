---
title: Root-State Trajectory Owner-Set Admission Census
description: A read-only census of whether the frozen Source-at-sixteen-rows plus sixteen-sample panel contains genuinely multiple-positive, strict physical-owner-set supervision for a 256-image training screen.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-23-trajectory-owner-set-admission-census
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_training_promotion_rejected
updated: 2026-07-23
---

# Root-State Trajectory Owner-Set Admission Census

## Decision and Outcome

The census is complete. Only 8 of the 2,004 Source-eligible training images
pass the frozen primary predicate, versus 256 required. Training promotion is
therefore rejected. The result establishes structural scarcity inside the 378
fully adjudicable images while retaining censoring as an unresolved
population-wide alternative because 1,626 images contain excluded candidates.
No single causal failure diagnosis is chosen. See [results.md](results.md) for
the complete evidence, claim boundary, artifact identities, and independent
review gate.

This unit decides whether the existing frozen trajectory data can support a
256-image, order-independent final physical-owner-set training screen. It does
not authorize training, freeze a trajectory score, select a model architecture,
or reinterpret a failed Source route as an empty owner set.

The decision-owning outcome is the number and composition of training-split
images whose root-state candidate group contains both:

1. at least one admissible strict trusted-owner-set inclusion edge; and
2. genuinely multiple positive serializations with different valid first-owner
   identities, rather than one canonical maximal trajectory.

The unit of analysis is one image at the exact shared root prompt. Each image
has one bounded Source continuation and sixteen sampled continuations. `B16`
is an exact token projection, not a post-parse row filter. Every sampled
continuation must contain exactly one terminal `<|im_end|>` token (`151645`),
it must be the final generated token, and it is excluded from the projected
token sequence and projected-token hash. Complete rows are scanned in
generation order through `<|box_end|>` token `151649`. If a route reaches
sixteen complete rows before any malformed or dropped span, its projection is
the exact token prefix ending immediately after the sixteenth row; later
tokens, malformed spans, and drops are outside `B16`. If natural end occurs
first with fewer than sixteen complete rows and no earlier malformed or
dropped span, the projection is the complete pre-`<|im_end|>` sequence. Any
malformed or dropped span before the sixteenth row, or before an earlier
natural end, gives status `failed_invalid_before_budget`; the candidate remains
diagnostic-only and may not be repaired by deleting the invalid span. Alias
deduplication uses projected token identifiers and their hash. The raw full-
completion identifiers and hash remain separate provenance.

## Falsifiable Question

Among the 2,004 Source-eligible members of the frozen 2,048-image training
split, do at least 256 images contain a nonzero, safely attributable strict
owner-set preference signal without treating an owner exchange, unknown row,
or singleton maximal serialization as order-independent supervision?

If not, the current automatically attributable evidence is insufficient for
the proposed 256-image treatment. The census must separate structural absence
among fully adjudicable candidates from censoring caused by unknown,
ambiguous, dropped, or geometry-untrusted rows. Only the former supports an
improved-exploration or separately declared order-alias branch. The latter
routes first to a train-only adjudication or matcher-coverage probe. Unique-row
or unique-trajectory cross-entropy is not an allowed fallback in either case.

## Competing Explanation

The working hypothesis is that low-temperature sampling has already produced
enough nested trusted owner sets and naturally diverse positive orders to train
the final-set partial order offline.

The strongest alternative is that sampled trajectories mostly exchange owners,
contain unresolved rows, or expose one maximal serialization per image. Under
that alternative, apparent owner diversity does not supply a valid
multiple-positive set objective even though sampled union coverage is large.

The separating observation is a complete root-state census over the frozen
training split, not a model run or a training loss.

## Frozen Source Boundary

The source model is the geometry-sorted, description-first Qwen3 Vision-
Language 2-billion-parameter checkpoint at step 4,887 used by the preceding
constant-dose screen. `Source@B16` means the first sixteen complete rows from
the frozen repetition-penalty-1.0 greedy realization, or all rows before an
earlier natural image end. `B16` expands to a budget of sixteen complete rows.

Inputs are immutable for this unit:

| Input | Frozen identity |
| --- | --- |
| Complete candidate pool | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/candidate-pool-v1/candidate-pool-2432.coord.jsonl`; Secure Hash Algorithm 256-bit checksum (`SHA-256`) `133afcf6659b78d71893e9f79e568dca676515a05c7fe23f3c237de01350b7e2` |
| Training-candidate split | `candidate-pool-v1/train-candidate.jsonl`; 2,048 images; SHA-256 `c0efce5806e0ee487298f7b3b7697f4ebe663d1fa210b2e0e10e08de08ab5746` |
| Split receipt | `candidate-pool-v1/split-receipt.json`; SHA-256 `478afe838c63dab7e55cad4ad16d8d1d0aa490231de8f25a0766768bda3678bf` |
| Sampled trajectories | `trajectory-panel-2432-vllm/production-v2/`; sixteen naturally closed trajectories per image, sample indices 0 through 15 |
| Bounded Source trajectories | `source-b16-vllm/production-v1/`; one frozen Source realization per image |

All relative artifact paths in the table resolve beneath:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
```

The two trajectory roots are frozen exactly, not merely by basename:

- sampled root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-2432-vllm/production-v2`;
  ordered manifest-set SHA-256
  `28304194b0b5e2aa7a718b2ee080253f616a8d9cf8fde8653e335589b3179b60`;
- Source root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/source-b16-vllm/production-v1`;
  ordered manifest-set SHA-256
  `4e979f920798dcd338c0620500d36b007f90404bb4a2852014702de93b9f521e`.

For each root, the ordered manifest set is the canonical JSON SHA-256 of the
eight records `{"relative_path": <root-relative POSIX path>, "sha256": <file
SHA-256>}`, ordered lexicographically by manifest path, with keys sorted and no
insignificant whitespace. Every bound batch artifact across both roots must
contain the same exact `model_identity.execution_model_identity` and
`model_identity.tokenizer_identity` objects. Their canonical JSON SHA-256
values are respectively
`8e0cc5c679df56cd55f965b56a4e9f4a661c0277bc93248100a0d143190918ac`
and
`878bc75fd27e4668788cb864bf93dee4d90dbbbbceec0f2037ba8eab75247fd1`.
A root, manifest-set, execution-model, or tokenizer mismatch fails the census
before semantic route inspection.

The split receipt must prove 2,048 training, 256 development, and 128 held-out
identifiers with pairwise zero overlap. Development, held-out, and the twelve
human-refined validation images are excluded before semantic route inspection.
Global manifest and file hashes may be verified, but the analyzer must filter
by image identity before reading stop, parser, text, owner, or other route-
derived fields. It must not compute or persist development or held-out route
summaries. Exactly 2,004 training identifiers must have an accepted
`Source@B16` status. The 44
Source-ineligible training identifiers remain ineligible and may not be
recollected, relabeled as empty-Source candidates, or used in any denominator
that implies a Source comparison.

Manifest inventory equality is exact, not count-only. Within each trajectory
root, manifest batch image identifiers must be duplicate-free and their union
must equal the complete 2,432-image candidate-pool identifier set. The sampled
manifest inventory, Source manifest inventory, and candidate-pool inventory
must therefore be identical sets; any missing, duplicated, or alien identifier
is a terminal input-contract failure even when all counts are 2,432. Record the
SHA-256 of the numerically ordered identifier array for each inventory in
`receipt.json`.

## Trusted Owner and Geometry Boundary

This census uses conservative annotation-anchored physical owners. A generated
row commits an owner only when the existing category-constrained global matcher
finds one unambiguous Common Objects in Context annotation at intersection over
union at least 0.50. A same-category row matched to an already committed owner
is a confirmed duplicate. Ambiguous matches, unmatched rows, semantic mismatch
candidates, and potentially unlabeled entities remain unresolved or unknown;
they are not hallucinations and receive no negative interpretation.

Entity credit and geometry adequacy remain separate. The census records the
0.50 annotation match used for conservative owner identity. Primary
whole-trajectory admission additionally requires every scored complete row,
including a confirmed duplicate, to have intersection over union at least 0.75
with its resolved owner. The 0.75 threshold is the existing trusted-geometry
surface; an entity match between 0.50 and 0.75 remains useful diagnostic owner
evidence but is not exact trusted geometry for this loss. The census must not
silently upgrade an entity match into exact trusted geometry.

A candidate is eligible for the primary partial order only when:

- its projected continuation has no malformed or dropped row before the
  sixteen-row budget;
- every evaluated complete row has a resolved owner or confirmed duplicate
  identity;
- every evaluated complete row has trusted geometry at intersection over union
  at least 0.75 with that resolved owner;
- no ambiguous, unmatched, unsupported, semantic-error, or otherwise unknown
  row remains; and
- its token, prompt, image, model, decode, parser, and Source-projection
  identities pass the frozen artifact contract.

Whole-candidate exclusion is mandatory. An unknown-bearing candidate remains a
diagnostic record and cannot be converted to an empty set, a harmful route, or
an indirect negative through a whole-trajectory comparison.

## Set and Alias Semantics

For each eligible candidate, record:

- final unique trusted-owner set;
- ordered sequence of first occurrences of trusted owner identities;
- first-owner identity;
- exact projected generated-token hash, with the raw full-completion hash kept
  separately;
- duplicate, malformed, confirmed-false, semantic-error, unknown, and
  premature-stop counts or indicators;
- parser and geometry-adequacy evidence; and
- decode mode and sample identity.

Deduplicate identical projected generated-token hashes before counting exact
serializations. Group the remaining candidates by exact final owner set. Two
candidates in the same group are outcome aliases; distinct ordered first-
occurrence sequences are row-order aliases.

For two outcome classes `A` and `B`, create a primary semantic edge only when
the owner set of `A` strictly contains the owner set of `B` and every exact-
token alias in `A` is no worse than every exact-token alias in `B` on each
frozen harm axis. Equivalently, the componentwise maximum harm in `A` must not
exceed the componentwise minimum harm in `B`. This universal rule is required
because the proposed objective compares every higher-set alias with every
lower-set alias. Also report existentially safe alias-pair counts as a
diagnostic, but do not use a minimum, mean, or one favorable alias to admit the
whole class edge. Owner-exchange sets are incomparable. Retain every strict
inclusion edge, including transitive edges; this unit does not select one pair.

An edge is admission-eligible only when every valid first-owner identity among
the lower-set aliases also appears as the first owner of at least one higher-set
alias. Otherwise the whole-trajectory preference could suppress a valid next
owner only because its observed route stopped early.

The census reports, without merging them into one favorable number:

- all valid outcome classes and all strict semantic edges;
- edge-admissible versus first-owner-orphaned edges;
- nondominated outcome classes;
- exact-token aliases, distinct row-order aliases, and distinct first-owner
  identities per nondominated class and across the positive frontier;
- images whose positive frontier has at least two exact serializations that
  begin with different owners;
- images with only one maximal exact serialization;
- incomparable owner-exchange pairs; and
- excluded candidates by parser, unknown-row, ambiguity, and geometry reason.

Call an image `fully adjudicable` only when all seventeen root-state
candidates pass primary candidate eligibility. Report structural edge and
alias scarcity both over fully adjudicable images and over all images with at
least two eligible candidates. Report the remaining images as censored by
their candidate-exclusion reasons; do not treat a censored image as evidence
that the missing strict relation does not exist.

The primary natural-alias predicate is frozen here. An image passes only when
at least one nondominated exact-owner-set class:

1. contains at least two unique generated-token serializations after exact-
   hash deduplication;
2. contains at least two distinct first-owner identities among those
   serializations; and
3. is the higher-set side of at least one universal-safety, first-owner-
   preserving admissible strict edge.

Frontier-wide diversity assembled from incomparable owner sets is diagnostic
only. The post-census training unit may choose a trajectory score but may not
replace this admission result with a looser alias definition.

## Reconstructable Analyzer Identity

A commit identifier, dirty-status digest, or analyzer-file hash alone is not
sufficient source provenance. Before publication, copy byte-for-byte task-
scoped source snapshots beneath `source-snapshot/` and write
`source-snapshot/manifest.json`. The minimum frozen paths are:

- `scripts/research/analyze_trajectory_owner_set_admission_census.py`;
- `scripts/research/assemble_constant_dose_breadth_state_banks.py`;
- `scripts/research/analyze_individual_trajectory_union_support.py`;
- `scripts/research/assemble_positive_path_imitation_state_bank.py`;
- `src/inference/backend.py`; and
- `src/config/fingerprint.py`.

Include any additional repository-local module whose code is executed
transitively by this census. For every snapshot, record the repository-relative
source path, source SHA-256, snapshot-relative path, and snapshot SHA-256, and
require the source and snapshot hashes to match. Record a canonical ordered
source-set digest in both the snapshot manifest and `receipt.json`. Git commit
and dirty-diff identity remain contextual evidence but do not substitute for
these reconstructable bytes.

## Required Census Outputs

The immutable execution root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-23-trajectory-owner-set-admission-census/production-v1/
```

A completed publication must contain:

- `image-census.jsonl`: one complete record for each of the 2,004 eligible
  training images, including every eligible and excluded candidate;
- `summary.json`: aggregate counts, histograms, admission variants, and the
  256-image feasibility result;
- `source-snapshot/` and `source-snapshot/manifest.json`: the reconstructable
  task-scoped source bytes and binding manifest;
- `receipt.json`: terminal status `completed`, source commit and dirty-diff
  identity, exact input roots and hashes, ordered manifest-set digests,
  candidate-pool inventory-equality evidence, shared execution-model and
  tokenizer identity digests, source-snapshot binding, row counts, integrity
  checks, determinism hashes, and output hashes; and
- no model checkpoints, training records, or gradient-bearing artifacts.

Publication is directory-atomic. Build every success artifact in a unique
sibling staging directory, write and verify `receipt.json` last, make the staged
tree immutable, and expose `production-v1/` only by one same-filesystem atomic
rename after all checks pass. On any exception, discard the success staging
tree and atomically publish a fresh failure-only directory at `production-v1/`
containing only `receipt.json` with terminal status `failed`, analyzer identity,
failed validation stage, and failure class and message. Readers accept census
outcomes only from a complete directory whose receipt says `completed` and
whose declared artifact hashes reproduce.

The analyzer may reuse
`scripts/research/assemble_constant_dose_breadth_state_banks.py` only as a
read-only input adapter. It must not call its StateBank writers or selection
logic.

## Acceptance, Branches, and Stop Rule

The census is mechanically accepted only if:

1. the split identities and hashes reproduce exactly;
2. all global artifact and manifest hashes match without deriving excluded-
   split route summaries;
3. the 2,048 training images each have sample indices 0 through 15 and all
   inspected training sampled routes naturally close;
4. the training split has exactly 2,048 Source records;
5. the training join contains exactly 2,004 accepted images and excludes the
   same 44 ineligible images;
6. every admitted candidate and edge is reproducible under reversed input
   order; and
7. all aggregate counts reconcile to the image-level records;
8. both manifest image inventories equal the exact candidate-pool identifier
   set;
9. both frozen roots, both ordered manifest-set digests, and the shared
   execution-model and tokenizer identity digests reproduce exactly;
10. sampled `B16` projection reproduces the frozen terminal-token, complete-
    row-boundary, and malformed-before-versus-after-budget semantics;
11. every task-scoped source snapshot reproduces its source hash and the
    source-set digest; and
12. the published directory is either one complete atomically published
    success set or one atomically published failure-only receipt, never a
    partial success set.

The data supports the planned 256-image screen only if at least 256 training
images pass the frozen primary natural-alias predicate. A result below 256
stops training promotion. If unknown, ambiguity, parser-drop, or geometry
exclusion censors potentially useful groups, the bounded conclusion is
`current attribution insufficient`, and the next probe is train-only review or
matcher coverage. If a sufficiently adjudicable population instead lacks
strict edges or same-set first-owner-diverse aliases, the bounded conclusion is
`structural signal insufficient`, and the next branch is improved exploration
or separately tested order-alias construction. The analyzer must report both
attrition and structural counts rather than choosing the favorable diagnosis.
A result at or above 256 permits design of a separate training unit; it does
not by itself choose a trajectory score or prove transfer to clean greedy
rollout.

After the census closes, invoke the independent advanced-model review gate only
if trajectory-score normalization, Source preservation, matched control, or
the meaning of the natural-alias counts remains conclusion-critical for the
training launch. Do not use external review instead of completing this census.

## Non-Goals and Cost

- no graphics-processing-unit work;
- no trajectory recollection or annotation mutation;
- no development, held-out, or manually refined image inspection for
  admission;
- no loss, StateBank, configuration, OpenSpec, or training implementation;
- no architecture or explicit task-state-carrier decision; and
- no claim that official annotations exhaust every visible physical owner.

Expected cost is one read-only artifact pass plus deterministic aggregation.
On success, stop after the completed receipt and bounded interpretation are
written. On failure, stop after the failure-only terminal receipt is atomically
published.

## Closure Contract

Close this unit in a separate `results.md` with `Observed`, `Supported`,
`Ruled out`, `Unresolved`, and `Not claimed`. Update the experiment index and
project memory only after the result changes the continuation route. The
constant-dose breadth unit remains immutable historical evidence.
