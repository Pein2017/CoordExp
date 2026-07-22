---
title: Source-Route Preservation and Multiple-Sampled-Route Training Screen
description: A 256-image treatment screen that tests whether replaying several safe sampled routes can expand greedy physical-owner coverage when ordinary Source-model routes are explicitly retained.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-22
---

# Source-Route Preservation and Multiple-Sampled-Route Training Screen

## Question

Can exact-prefix training on several safe, complementary sampled routes make
the frozen Source model recover more unique physical objects under ordinary
greedy decoding, without exchanging away the ordinary owners that Source
already finds?

The experiment tests a loss-only treatment using Qwen3-VL's own complete row
trajectories. It does not add an object slot, detector, covered-set memory,
inference controller, or terminal suppression rule.

## Why This Treatment Follows from the Evidence

The earlier one-route treatment moved route-added owners into greedy output but
lost nearly as many ordinary owners. On the 118 admitted training images, the
best milestone changed route-added-owner coverage from `93` to `109` while
ordinary-owner coverage changed from `712` to `697`; total owner coverage did
not expand. This showed that exact-prefix row supervision can move the model,
but one sampled path mainly replaces one enumeration habit with another.

The July 22 completion study independently found that prefix order changes
future owner sets even when the covered set and final row are fixed. Therefore,
one canonical route is not an adequate training target. It also found that
globally suppressing terminal output adds many rows but few owners, so merely
lengthening rollout is not the treatment.

## Feasibility Census Already Executed

The frozen 256-image collection contains one Source greedy route and sixteen
low-temperature sampled routes per image. Under the existing conservative
route admission rule:

- 122 images have at least one parser-accepted, naturally closed sampled route
  whose verified-owner set strictly contains the greedy set without more
  confirmed duplicates or malformed rows;
- 553 sampled routes pass that rule;
- 91 images have at least two safe routes;
- 61 images have at least two different added-owner sets;
- the safe-route union contains 298 route-added physical owners across the 122
  images;
- 119 of the 122 images also contain at least one trusted Source greedy row,
  giving 767 possible Source-route preservation anchors.

A stricter alternative was also counted: multiple different owners observed
under one identical token-exact prefix. Only six images provide such groups,
and every group is at row zero. That surface is too sparse and too shallow to
serve as the main 256-image training signal. It remains a mechanism subset,
not the primary treatment.

This census is inherited evidence from the immutable trajectory artifacts. It
does not yet constitute treatment evidence.

## Competing Explanations

### Working explanation: multiple routes plus preservation can expand support

Different sampled routes expose complementary object modes. Replaying several
safe routes teaches more than one useful path, while replaying ordinary Source
rows prevents the update from paying for new owners by erasing old ones.

### Strong alternative: positive row replay only changes the preferred route

Even with multiple routes and Source rehearsal, language-tower parameter
updates may still move probability among a roughly fixed set of output slots.
Training likelihood may fall while clean greedy coverage stays flat, ordinary
owners disappear, or output becomes longer without discovering new owners.

The gained-versus-lost owner ledger and the non-admitted-image subgroup are the
primary controls separating these explanations.

## Terminology

- **Source checkpoint**: the geometry-sorted, description-first,
  pure-cross-entropy plus token-type-gate Weight-Decomposed Low-Rank Adaptation
  checkpoint at step 4,887.
- **Source anchor**: the first trusted occurrence of one physical owner in the
  Source greedy route, replayed under its exact Source-generated prefix.
- **Safe sampled route**: a parser-accepted, naturally closed sampled route
  that strictly expands the Source greedy verified-owner set at a fixed
  sixteen-row budget without more confirmed duplicates or malformed rows.
- **Treatment row**: a trusted first-occurrence row from a selected safe
  sampled route, replayed under its exact sampled prefix.
- **Physical owner**: one reviewed or annotation-backed object instance,
  distinct from a category label or a row string.
- **Source preservation**: rehearsal of exact trusted Source rows. It is a
  positive replay constraint, not a proof that every Source behavior is fixed.
- **StateBank**: the repository's immutable event store that binds an image,
  exact prompt and prefix token identifiers, candidate-row token identifiers,
  physical-owner review, selected gradient sites, and source-checkpoint
  identity for replay training.

## Frozen Data Scope

Use the existing 256-image training collection:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-earliest-shared-prefix-branch-and-trajectory-treatment/
full-trajectory-k16-source-256/
```

The twelve human-refined validation images are development and safety evidence
only. They must never enter a StateBank used for gradients.

Use 118 admitted physical images. Start from the previous one-route cohort,
exclude image `319139` because enlarged-crop review could not assign any greedy
row to one unique physical owner, and admit image `35514` as its deterministic
replacement. The remaining 138 images in the 256-image collection form a
non-admitted transfer subgroup.

### Source anchor admission

Keep only the first Source greedy row assigned to a trusted physical owner.
Malformed, unmatched, duplicate, and unresolved rows receive no gradient.
Schema and description tokens may be retained when owner identity is trusted.
Coordinate tokens are retained only when geometry passes the existing trust
rule; otherwise coordinates are exact context with zero coordinate gradient.

### Single-route treatment admission

Reuse the 505 unique-provenance treatment events that remain after removing
image `319139`, add the six valid rows from replacement image `35514`, and trim
fifteen events deterministically to retain exactly 496 sampled events across
all 118 images. Do not clone events or supplement the arm from a second sampled
route merely to round the batch.

### Multiple-route treatment admission

For each image, select at most three safe sampled routes. Selection is
deterministic and prioritizes:

1. marginal route-added physical owners not already supplied by a selected
   route;
2. total added owners;
3. fewer unresolved rows;
4. lower seed identifier.

From each selected route, retain trusted first-occurrence rows through the last
owner that contributes marginal coverage. Deduplicate identical
`(image, exact prefix, exact row, physical owner)` events. Select a matched 496
treatment events across the same 118 images while preserving image diversity
and marginal-owner diversity.

One exact sampled event is an explicit geometry exception: image `314812`,
route `seed-31002`, row 2, owner `314812:1986481` (`book`). Enlarged-crop review
supports the physical entity and category but not the thin predicted extent.
Retain schema and description supervision for this row and mask all four
coordinate sites. This does not relax geometry admission for any other row.

### Matched Source preservation set

Select exactly 496 Source anchors from the same 118 images, deterministically
and with at least one anchor per image. Use the identical Source-anchor set in
both treatment arms.

## Compared Arms

### Source checkpoint

Evaluation only; no additional training.

### One safe route plus Source preservation

Combine 496 single-route treatment events with the 496 matched Source anchors.

### Multiple safe routes plus Source preservation

Combine 496 rows selected across at most three complementary safe routes per
image with the same 496 Source anchors.

The completed one-safe-route-without-preservation checkpoint remains historical
context. It is not rerun and is not treated as a perfectly matched causal arm.

## Training Objective

Every event contains one exact complete object row `Y` and its exact generated
prefix `P`. The prefix is context only. For row tokens, compute separate mean
negative log probabilities for:

1. schema and description sites; and
2. trusted coordinate sites.

Average the non-empty groups. Apply the existing token-type gate at the same
selected sites as a language-format stabilizer.

For image `i`, split total image credit equally between the Source and sampled
families. If family `f` has `n_i,f` events, each event receives unnormalized
weight:

```text
w(i, f, event) = 1 / (2 * n_i,f)
```

Rescale all event weights to mean one within an arm. Thus route count, row
count, description length, and scene density do not increase one image's total
training credit. The objective does not force equal next-owner probabilities;
it only presents several successful paths instead of one.

## Model and Optimization Scope

- Freeze the vision tower, multimodal aligner, and selected-token embedding
  deltas.
- Train only the language-tower Weight-Decomposed Low-Rank Adaptation payload.
- Use learning rate `1e-5`, gradient clipping at `1.0`, and effective event
  batch size `32`.
- Use eight Graphics Processing Units. Under the current training runtime,
  effective batch size `32` resolves to four segment-isolated event
  micro-steps per rank before each optimizer update; it is not a tensor batch
  of four independent examples in one forward pass.
- Each arm contains exactly 992 unique-provenance events and runs one complete
  epoch: 31 optimizer updates.
- Save checkpoints at approximately 30, 60, and 90 percent of the epoch and at
  the final update. The live schedule resolver maps these to steps 10, 20, 30,
  and 31 for a 31-step run.
- Use the existing positive-path-imitation training profile, exact-token
  StateBank replay, segment-isolated packing, 32-bit floating-point loss math,
  Accelerate runtime, checkpoint writer, and ordinary inference path.
- Do not add canonical supervised-fine-tuning data, Kullback-Leibler
  divergence, terminal suppression, negative unmatched rows, online refresh,
  an external teacher, or a new inference-time component.

## Primary Observation

The primary observation is ordinary clean greedy physical-owner coverage, not
training loss or fixed-prefix row likelihood.

For Source and every saved milestone, report on the full frozen 256-image
collection and separately on the admitted 118 and non-admitted 138 images:

- total unique matched physical owners;
- route-added owners gained;
- ordinary Source owners retained and lost;
- gained-to-lost owner balance;
- prediction-row count and natural closure;
- confirmed duplicates;
- malformed, dropped, invalid, truncated, and repetition-burst rows;
- category discovery;
- common-owner box intersection over union, center error, size error, and each
  coordinate boundary.

Also evaluate the twelve human-refined images as a manual entity-versus-geometry
safety panel. Unmatched Common Objects in Context predictions are not automatic
hallucinations and remain review-needed evidence.

## Interpretation

- **New owners rise and ordinary owners are retained:** Source preservation
  repairs the owner-exchange failure; the treatment is a candidate for a
  1,024-image replication.
- **The multiple-route arm exceeds the one-route arm:** complementary route
  support is useful, not merely another pass over the same best path.
- **Both preservation arms retain owners but add none:** rehearsal stabilizes
  the model, but positive-only route learning still cannot expand greedy set
  coverage.
- **New owners and lost owners rise together:** the model still exchanges route
  families; do not promote unchanged.
- **Rows rise without unique owners:** the update mainly changes continuation.
- **Entity discovery improves but geometry worsens:** retain a bounded
  selection result, but do not claim complete detection improvement.
- **Only replay loss improves:** the treatment does not transfer through
  self-generated prefixes.

## Stop and Promotion Rule

No fixed metric margin is imposed before observing the screen. Do not discard
the evidence merely because it misses an expected effect. Promotion to 1,024
images is allowed only when at least one clean-rollout milestone is genuinely
promising after inspecting gained and lost owners, non-admitted transfer,
duplicates, malformed output, and geometry. A lower training loss, longer
rollout, or selected-route imitation alone is insufficient.

If both preservation arms fail in the same owner-exchange pattern, stop this
positive-only family. The next treatment must then compare valid uncovered and
harmful branches at exact self-prefix decision points or add a compact task
state; do not enlarge this objective unchanged.

## Minimal Execution Path

1. Assemble the frozen Source-anchor set and the two matched 992-event
   StateBanks with explicit selection receipts.
2. Run focused schema, exact-token, family-weight, image-balance, and blind-set
   checks.
3. Run one real one-step smoke for each bank; verify finite gradients,
   checkpoint save/load, and ordinary clean inference.
4. Train both 31-update arms on eight Graphics Processing Units.
5. Run matched clean inference and owner-ledger comparison for Source and all
   milestones.
6. Inspect the highest-value gained/lost cases and the twelve human-refined
   images before deciding whether to promote to 1,024.

## Non-Goals

- claiming a final architecture or complete mechanism;
- making all remaining owners equally probable;
- forcing one canonical object order;
- treating every sampled row as positive;
- treating unmatched predictions as hallucinations or negatives;
- using the twelve human-refined images for gradients;
- fixing box ambiguity with exact-coordinate supervision in this screen;
- full-dataset training before the 256-image result closes.

## Artifact Handle

Logical output root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/<run-id>/
```

Every execution run identifier is immutable. The initial feasibility census is
locally verified against the inherited 256-image trajectory artifact. The
first immutable run, `state-banks-v1`, was rejected before training because
it contained cross-family semantic duplicates and did not prove rollout
checkpoint identity from runtime evidence. The corrected 992-event StateBanks
exist under immutable run identifier `state-banks-v2`. A read-only launch gate
independently approved exact provenance, 496-plus-496 family balance, all 992
semantic identities, matched Source rows, image-family weights,
runtime-derived checkpoint identity, blind-set exclusion, and geometry
masking. The real smoke gate also passed: one greedy Source event completed a
finite one-step update, and one eight-rank mixed step consumed exactly sixteen
sampled-route plus sixteen Source-preservation events with finite identical
all-rank gradients and a saved checkpoint. That mixed checkpoint loaded through
the ordinary Hugging Face inference path and completed all twelve human-refined
rows with zero parser or score failures. Two rows stopped at the 512-token smoke
limit, so this load test is not benchmark evidence. Formal training and
matched comparison receipts now exist. The decision-grade interpretation is
recorded in [results.md](results.md). The screen did not promote this objective
to a 1,024-image training run. It established a real, target-specific learning
effect, but that effect remained concentrated on selected owners and admitted
images while other owners were exchanged away.
