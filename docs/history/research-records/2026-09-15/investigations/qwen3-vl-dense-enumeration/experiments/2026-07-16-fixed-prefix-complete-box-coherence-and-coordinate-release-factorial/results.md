---
title: Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-Release Results
description: Verified two-target evidence separating local boundary continuation, cross-coordinate geometry transport, part-sized late-extent basins, and cross-row transition effects.
type: investigation-result
role: research-evidence
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial
status: complete
evidence_status: verified
updated: 2026-07-17
---

# Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-Release Results

## Scope

This unit tests two manually adjudicated, exact pre-`x1` states from the
Qwen3 Vision-Language (`Qwen3-VL`) 2-billion-parameter model with the
geometry-sorted step-4887 adapter. It is a purposive mechanism case study, not
a population estimate and not an architecture promotion.

The completed panels are teacher-forced full-box scoring and causal progressive
coordinate release. The result is bounded to these two states and does not
promote an architecture or training objective.

## Frozen Inputs

Immutable fixture:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/
fixture-v5-20260716e/target-fixture.json
```

Fixture Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
cbf9040a4237f58c5365816b4b1e08b932bd18aa66f68515e114826f7ebb8f7a
```

The sidecar receipt binds the same fixture digest. Both score runs verified
the digest before model loading, used physical batch size one and repetition
penalty `1.0`, and reconstructed prompt hashes exactly equal to the frozen
pre-`x1` hashes.

Runtime discovery used the normal brain floating-point 16-bit (`bfloat16`)
model branch. All selected logits, log probabilities, and aggregate arithmetic
were stored or calculated in 32-bit floating point. The only close ordering is
explicitly held below rather than promoted from `bfloat16`.

## Panel 1: Complete-Box Teacher-Forced Likelihood Surface

Primary score:

```text
sum of four selected full-vocabulary coordinate-token log probabilities
```

The natural `<|box_end|>` log probability was excluded from the primary score
and was negligible. Coordinate-only normalization differed from the primary
score by only approximately `0.002` to `0.006` natural-log units.

### Target A: fork part versus Common Objects in Context visible whole

Artifact:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/
teacher-forced-score-target-a-fixture-v5-bf16-batch1-20260716c/
scores-target-A.json
```

Artifact `SHA-256`:

```text
276c9390e3564fba79a25f9f202caadb243ad4e160dd7d24387d54a7e3356915
```

| Path | Normalized box | Primary score |
|---|---:|---:|
| reviewed fork-head part and exact source-native row | `[658,450,780,487]` | `-9.979124` |
| Common Objects in Context visible whole fork | `[660,448,999,488]` | `-12.422686` |
| best non-endpoint hybrid | `[660,450,780,487]` | `-10.016030` |

The exact prefix favors the sampled part-sized path over the visible-whole
fork by `2.443562` natural-log units. The frozen mean coherence contrast is
`+0.601276`, but `11/14` hybrids outrank the weak visible-whole endpoint and
the best hybrid is only `0.036906` below the part path.

This is evidence for a prefix-local part-or-extent basin. It is not independent
native calibration because the source-native row is exactly the part path. It
does not yet distinguish a discriminative object-part shortcut from trajectory
selection, a right-image-edge prior, or a large-width penalty. Target A admits
no instance-owner claim.

### Target C: dense same-category chair field

Artifact:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/
teacher-forced-score-target-c-fixture-v5-bf16-batch1-20260716c/
scores-target-C.json
```

Artifact `SHA-256`:

```text
74bd1b1565c5e1f40b6cd51c05fc5acd9fb9e25d7831c6ba96a9079863a56c37
```

| Path | Normalized box | Primary score | Margin below target |
|---|---:|---:|---:|
| accepted lower-row target chair | `[537,121,651,349]` | `-13.945811` | `0.000000` |
| exact source-native row | `[552,123,660,357]` | `-14.228432` | `0.282621` |
| adjacent same-row chair | `[643,126,730,350]` | `-19.343024` | `5.397213` |
| exact cross-row union | `[519,33,657,347]` | `-30.969144` | `17.023333` |
| upper-row chair | `[557,30,656,134]` | `-37.685971` | `23.740160` |

The highest-scoring paths form a target-like lower-row ridge. The best path,
`[537,121,657,349]`, preserves target `x1`, `y1`, and `y2` while moving `x2`
six bins toward the source-native row. It exceeds the accepted target by only
`0.308436` natural-log units. The upper-row `y1` and `y2`, adjacent-chair
`x1`, and union `y1` are strongly incompatible with this exact prefix.

The frozen mean coherence contrasts are:

| Pair | Coherence contrast |
|---|---:|
| target versus adjacent chair | `+4.598258` |
| target versus upper-row chair | `+0.765957` |
| target versus cross-row union | `+0.312384` |

These numbers are not instance-owner evidence. In every chair panel, `9` to
`13` of the `14` non-endpoint hybrids outrank the weak alternate endpoint.
The large target-versus-alternate margins show conditional target-region
compatibility; they do not show that an early coordinate causally binds later
coordinates. The `0.081` to `0.308` natural-log ordering among target-like
`x2` variants is treated as a near tie unless a full-model 32-bit floating
point run is ever needed to make that ordering conclusion-changing.

## Panel 2: Progressive Coordinate Release

All `128` scientific sampled suffixes and all `20` greedy wiring suffixes
generated a natural box closure. Exact force-four controls reproduced their
reference endpoints. The artifact classifier preserves two different axes:

- **boundary-configuration attribution** compares the full endpoint, hybrid,
  union, and same-category-region lattice;
- **endpoint-family attribution** compares only the coherent endpoint pair and
  is owner-compatible geometry evidence, not owner proof.

### Target C: strong horizontal cross-coordinate transport

Artifacts and Secure Hash Algorithm 256-bit (`SHA-256`) digests:

| Artifact | Digest |
|---|---|
| `progressive-release-target-c-adjacent-greedy-v2-fixture-v5-20260716c/release.json` | `d0a184e91ca3f31527d5bb8bcfe9f2d58d0ad84c1dd1845a2587e2eb500aaf7c` |
| `progressive-release-target-c-free-sampled16-v2-fixture-v5-20260716d/release.json` | `09845deba4afde0b0a038f6d2e680246b8482ae7edda0a0c5c351e912d214988` |
| `progressive-release-target-c-owner-force12-sampled16-v2-fixture-v5-20260716d/release.json` | `6d2157a2116efb713df20cb33bcc7162a8587a0fc0285596aa2e14a70b3c40f2` |

The free arm is target-like on all `16/16` suffixes. The owner-discriminative
slots for this pair are horizontal `x1` and `x2`; the vertical endpoint
differences are too small to own an instance claim.

| Forced prefix | Released `x2` range | Mean released `x2` | Endpoint-family result |
|---|---:|---:|---:|
| none | `652..689` | `669.7500` | target-like `16/16` |
| target `x1` | `651..687` | `673.3125` | target-like `16/16` |
| target `x1,y1` | `656..693` | `673.8750` | target-like `16/16` |
| adjacent `x1` | `741..781` | `755.0000` | adjacent-like `16/16` |
| adjacent `x1,y1` | `736..759` | `750.0625` | adjacent-like `16/16` |

Across the `64` paired force-one and force-two suffixes, released owner-
discriminative `x2` is closer to the cued endpoint on `63` and crosses to the
other endpoint on `1`. The target and adjacent ranges do not overlap.

This establishes a causal `x1 -> x2` transport effect at this exact state. It
falsifies literal boundary independence and a model in which `x2` ignores the
earlier horizontal decision. It does **not** distinguish:

```text
physical-instance owner state
versus
autoregressive box-width, shape-validity, or traversal-rank continuation
```

Changing `x1` also changes the geometrically valid support of `x2`. The full-
lattice classifier labels many samples as hybrids because it contains near-
endpoint one-bin variants; that descriptive label does not override the
disjoint endpoint-family result.

### Target A: early left and top boundaries do not recover full extent

Artifacts and `SHA-256` digests:

| Artifact | Digest |
|---|---|
| `progressive-release-target-a-part-whole-greedy-v2-fixture-v5-20260716e/release.json` | `59fabaf0e7d80ba3dd4a6a38b069cee1f52af71b99592acc2468e4634a21c51d` |
| `progressive-release-target-a-free-sampled16-v2-fixture-v5-20260716f/release.json` | `40c62fb39d5066e23d87e61475fe8dae66ea397aae398221dcb1c1aa7e66838a` |
| `progressive-release-target-a-part-whole-force2-sampled16-v2-fixture-v5-20260716f/release.json` | `035ec62f35641abde2fd590a2cfc6ef934c782ee5fe5c6a780e5ae0f43f35485` |

The free sampled arm is part-like on `15/16` suffixes. Its native `x2` lies in
`762..999`, with `15/16` values in `762..776` and one right-edge outlier.

After force-two, the endpoint-family counts are:

| Cue path | Part-like | Visible-whole-like | Outside both |
|---|---:|---:|---:|
| fork-head part `x1,y1` | `13/16` | `2/16` | `1/16` |
| visible-whole fork `x1,y1` | `14/16` | `2/16` | `0/16` |
| combined | `27/32` | `4/32` | `1/32` |

The paired outcome is stronger than the marginal count. Although the two
paths force slightly different `x1,y1`, their released `x2` is exactly equal
on `14/16` paired seeds; it is larger for the whole cue once and smaller once.
Their released `y2` is exactly equal on `13/16` seeds. Most continuations land
near `x2=763..781`, far before the visible-whole right boundary at `999`.

Therefore the early left/top cue is not sufficient to specify the complete
fork extent. The result supports a late part/extent basin and is inconsistent
with a simple model in which a complete fork owner was already selected and
only needed a plausible `x1,y1` anchor. It does not prove that the visual tower
cannot represent the shaft or that every utensil uses a part shortcut.

## Exploratory Cross-Row Observation

The sampler generated beyond the current row, so its trailing trajectory
supplies an unpreregistered but useful observation. The first description
after the current chair row was:

| Current-row condition | Next description |
|---|---|
| free | `person` `14/16`; `chair` `2/16` |
| target `x1` | `person` `16/16` |
| target `x1,y1` | `person` `13/16`; `chair` `3/16` |
| adjacent `x1` | `chair` `16/16` |
| adjacent `x1,y1` | `chair` `16/16` |

Thus the emitted current-row geometry causally changes the next textual
trajectory; the row is not simply forgotten after `<|box_end|>`. The adjacent-
cued next chairs usually occupy approximately `x=400..555`, a region already
represented by multiple chair rows in the prefix. Without a unique physical-
entity ledger, this is transition-state sensitivity with possible duplicate
re-entry, not uncovered-object redistribution.

## Hypothesis Adjudication

### Boundary-wise composition

**Narrowed, not accepted as literal independence.** Hybrid configurations are
frequent and the fork cue does not lock complete extent, supporting staged
boundary competition. However, the chair `x1 -> x2` intervention shows strong
causal dependence between horizontal coordinates.

### Shared instance owner with modal-versus-amodal extent uncertainty

**Not established.** Chair endpoint-family geometry switches coherently after
`x1`, but the design cannot separate a physical owner from box-shape and
traversal continuation. Fork early-boundary forcing fails to recover the
reportable whole extent. Shared owner state remains possible in selected
states, but it is not a sufficient universal explanation.

### Discriminative object-part shortcut

**Plausible but not isolated for Target A.** A part-sized or late-extent basin
is strongly supported: teacher-forced probability and sampled release favor
the fork-head extent, and early full-object left/top forcing does not recover
the visible shaft. The design does not separate discriminative-part
recognition from prefix trajectory, right-image-edge, or large-width priors.
This is a case-level observation, not a population prevalence claim.

### Same-category region aggregation

**Plausible but not isolated for Target C.** The prefix supports one local
target-like chair basin, while adjacent forcing produces another chair and
changes the following traversal. The result may reflect instance selection,
geometry-sorted rank, or a repeated-category region field.

## Bounded Verdict

The two states do not support one universal four-coordinate object file.

```text
description-fixed pre-x1 state
  -> choose an early spatial boundary
  -> autoregressively transport local box geometry
  -> decide late extent from a part/object/shape basin
  -> write the emitted geometry into the next textual traversal state
```

The experiment rules out literal coordinate independence, but it also rejects
the stronger claim that choosing an early plausible boundary necessarily locks
one complete physical object. The fork state strongly supports a local part-
sized or late-extent basin. The chair state strongly supports coordinate-
conditioned geometry transport and cross-row state influence, but not a proven
physical-instance owner or correct covered-set update.

## Numerical Decision

Runtime discovery used the normal brain floating-point 16-bit (`bfloat16`)
model branch. All selected logits, log probabilities, and score arithmetic
were stored or calculated in 32-bit floating point. Teacher-forced scoring and
greedy wiring used physical batch size one. Scientific sampled runs used
physical batch size four under the frozen sampling attestation. The chair
coupling reproduces qualitatively in the batch-one greedy controls, where all
`4/4` eligible released horizontal edges follow their cue; the `63/64`
distributional result is explicitly batch-four evidence. Full-model 32-bit
floating point was not escalated because no bounded mechanism verdict depends
on a near tie:

- the fork part-versus-whole score margin is `2.443562` natural-log units;
- the target-cued and adjacent-cued chair `x2` ranges are disjoint by at least
  `43` coordinate bins;
- the only `0.081` to `0.308` natural-log target-like chair ridge is explicitly
  retained as a near tie and does not select the conclusion.

## Stop-Rule Application

1. Targets A and C already separate clean part-extent behavior from crowded
   same-category spatial coupling, so Target B is not run.
2. The fork evidence is no longer ambiguous enough to justify more sampling.
3. The chair effect is large enough that a full-model 32-bit floating-point
   rerun cannot change the bounded causal-coupling conclusion.
4. The remaining distinction—physical owner versus spatial grammar—requires a
   different matched intervention, not more samples of the same arms.
5. No architecture or training inference is admitted.

## Most Valuable Next Discriminator

If this branch resumes, compare at the same fixed state:

```text
real-object x1 cue
versus
matched synthetic or object-free x1 cue with the same displacement and box-width prior
```

Score both the released current box and whether the next row is a new physical
object, a previously committed object, or a geometry-sorted recovery. A real-
object-specific effect would support instance binding; equal effects would
favor coordinate grammar or traversal-rank state. The successor must freeze a
unique physical-entity ledger before claiming uncovered redistribution.

## Verification

- fixture digest, source call, source row, image identity, and exact prompt
  reconstruction passed before model execution;
- teacher-forced score and greedy-control runs used physical batch size one;
- sampled runs used physical batch size four, and repetition penalty `1.0` was
  used throughout;
- paired seeds were shared across competing sampled paths;
- exact force-four endpoints and all natural closures passed;
- raw token identifiers, coordinate boxes, per-slot endpoint errors, endpoint-
  family classifications, and full-lattice classifications remain in the
  artifacts;
- no training, architecture change, object-slot system, or covered-set carrier
  was started or promoted.
