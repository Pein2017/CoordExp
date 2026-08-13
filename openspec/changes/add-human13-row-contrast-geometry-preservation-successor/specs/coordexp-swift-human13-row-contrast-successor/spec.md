## Purpose

Defines the bounded experiment-local behavior for redirecting duplicate-row
probability toward uncovered native owners, enforcing rectangle-valid greedy
coordinate decisions, and testing first-order preservation of frozen Source
owner coordinates on the exact Human-13 overfit panel.

## ADDED Requirements

### Requirement: Content-Addressed Successor Ledger

The successor SHALL bind the sealed Human-13 manifest, Source identity,
selected native rows, exact duplicate prefixes and complete duplicate rows,
covered and uncovered `G union H` owner sets, candidate aliases, coordinate
sites, and optional authoritative A4 exposure-one/two negative events in one
canonical content-addressed ledger. K-miss owners SHALL remain gradient-neutral.

Optional A4 events SHALL be admitted only after exact artifact, tokenizer,
parser, token-span, and image identity validation. Failure to align one A4
artifact SHALL produce a typed exclusion and SHALL NOT alter the sealed
Source/K event family.

#### Scenario: Exact successor ledger is materialized
- **WHEN** the sealed manifest and declared prior-output artifacts match their frozen identities
- **THEN** the ledger SHALL contain deterministic complete-row events, owner alternatives, rectangle sites, and `G` watch sites
- **AND** repeated identical image/prefix/duplicate-row events SHALL have one canonical identity.

#### Scenario: Prior output cannot be aligned exactly
- **WHEN** a declared A4 output row lacks exact tokenizer/parser span alignment
- **THEN** that prior-output event SHALL be excluded with a typed reason
- **AND** no approximate token position SHALL enter training.

### Requirement: Hierarchical Duplicate-Row Contrast

Every admitted duplicate event SHALL use one deterministic contrast branch.
Same-description uncovered owners SHALL use owner-normalized four-coordinate
row scores; otherwise uncovered owners SHALL use owner-normalized complete
owner-distinguishing row scores. If no trusted uncovered owner exists, all four
duplicate coordinate sites SHALL receive stable unlikelihood and no terminal
target SHALL be introduced.

Candidate aliases SHALL be normalized within owner before owners are combined.
Every event SHALL be consumed, with event losses averaged within image and then
across eligible images. All score reductions SHALL execute in fp32.

#### Scenario: Same-description uncovered owner exists
- **WHEN** a duplicate event has at least one uncovered valid owner with the same normalized description
- **THEN** its bbox score SHALL be contrasted with owner-normalized valid bbox alternatives
- **AND** shared description/schema tokens SHALL NOT be treated as duplicate negatives.

#### Scenario: Only different-description owners remain
- **WHEN** no same-description uncovered owner exists but another trusted uncovered owner does
- **THEN** the duplicate's owner-distinguishing row score SHALL be contrasted with owner-normalized valid complete-row alternatives.

#### Scenario: No trusted uncovered owner remains
- **WHEN** the event has no trusted uncovered `G union H` candidate
- **THEN** all four duplicate coordinate tokens SHALL receive stable unlikelihood
- **AND** `STOP` SHALL NOT be used as a positive or redirect target.

### Requirement: Rectangle-Valid Greedy Gate

At every declared positive-row `x2` and `y2` site, the successor SHALL compare
the maximum logit among coordinate tokens that make the corresponding extent
strictly positive with the maximum logit among all invalid vocabulary tokens.
It SHALL apply a fp32 margin hinge with detached selector identities.

The gate SHALL accept any coordinate token in the valid set; it SHALL NOT
require the canonical coordinate to be top one or enumerate complete GT-IoU
boxes.

#### Scenario: Invalid coordinate is current argmax
- **WHEN** the global top token at `x2` is non-coordinate or has value no greater than emitted `x1`
- **THEN** the loss SHALL move the highest rectangle-valid coordinate above that invalid competitor by the frozen margin.

#### Scenario: Another valid coordinate is preferred
- **WHEN** a non-canonical coordinate is the highest token and still makes a positive extent
- **THEN** that token SHALL be accepted as valid
- **AND** the gate SHALL contribute zero once the valid margin is satisfied.

### Requirement: Frozen-G First-Order Gradient Preservation

R2 SHALL separately accumulate the complete R1 gradient and the exact frozen
Source-`G` coordinate watch gradient at one unchanged parameter state. When
their dot product is negative, R2 SHALL remove the conflicting component before
global gradient clipping and AdamW. When it is nonnegative, R2 SHALL leave the
R1 gradient unchanged.

The receipt SHALL report pre/post dot products, gradient norms, projection
coefficient, finite status, and zero-denominator disposition. The requirement
is limited to the raw trainable-parameter gradient and SHALL NOT claim that
AdamW or free-running decode preserves every owner.

#### Scenario: R1 conflicts with the G watch direction
- **WHEN** the accumulated trainable-parameter dot product is finite and negative
- **THEN** R2 SHALL project the R1 gradient so the post-projection dot product is nonnegative within numerical tolerance.

#### Scenario: Watch gradient is unusable
- **WHEN** the watch gradient is missing, non-finite, or has norm below the frozen epsilon
- **THEN** R2 SHALL fail before clipping, optimizer mutation, or checkpoint publication.

### Requirement: Bounded Two-Arm Execution and Evaluation

The successor SHALL run only R1 and R2 from byte-identical Source payloads and
fresh AdamW states, at cumulative exposures one and two. It SHALL reuse the
historical A4 exposure-two output without rerunning A4. Training SHALL remain
world-size one with no-padding isolated packing and one optimizer update per
complete panel exposure.

Before the full two-arm execution, image 14038 SHALL complete one real R1
materialize-to-analyze vertical slice. Every evaluated checkpoint SHALL use HF
fp32/SDPA, physical batch one, the original prompt, clean greedy decoding, and
repetition penalty `1.0`.

#### Scenario: Production-shaped vertical slice succeeds
- **WHEN** image 14038 completes one R1 exposure
- **THEN** the run SHALL publish a readable checkpoint and one analyzer-compatible clean-greedy output
- **AND** the receipt SHALL bind the same ledger, trainable surface, packing, checkpoint, and evaluation identities used by the full run.

#### Scenario: Six-GPU critical path is launched
- **WHEN** the vertical slice and CPU gates pass
- **THEN** at most two independent training jobs and four independent checkpoint evaluations SHALL execute concurrently
- **AND** no extra arm or multi-rank training job SHALL be invented to occupy spare GPUs.

### Requirement: Same-Panel Outcome and Claim Boundary

The analyzer SHALL report owner-identity `H` gains, `G` losses, incidental `M`
gains, unique owners, duplicates, malformed rows, cap stops, rows, generated
tokens, and runtime without reducing them to one scalar. It SHALL expose
per-image gained/retained/lost identities and reuse the predecessor's duplicate
and one-to-one owner matching semantics.

The output SHALL be labeled same-panel overfit evidence only and SHALL NOT be
used for validation, transfer, checkpoint promotion, full-set, or production
claims.

#### Scenario: New H owner replaces a G owner
- **WHEN** pooled unique-owner count hides an H gain paired with a G loss
- **THEN** the analyzer SHALL report both identities and counts separately
- **AND** SHALL NOT label the outcome pure consolidation.
