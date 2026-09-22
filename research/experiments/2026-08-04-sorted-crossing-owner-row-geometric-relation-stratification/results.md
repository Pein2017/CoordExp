---
title: Sorted Crossing Owner-Row Geometric Relation Stratification - Results
description: Verified CPU-only geometry stratification of the crossing-row likelihood tail, with posthoc visual-support adjudication kept outside the frozen decision.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-04
---

# Results

## Verdict

The unit is complete and independently verified. Its frozen primary route is
`separated_competition_survives`.

Of the `26` primary P+C->E owner rows, `9` have an image-referenced coordinate
change at or below `-1.0` nat. Five of those nine are same-description rows
whose C/E boxes overlap or contain one another. Four are clearly separated
rows spanning three images, so the predeclared separated-first rule fires.
Three of the four separated rows also change description.

This result rejects a pure identity-slippage or healthy duplicate-suppression
account as a sufficient explanation. It does not identify owner-specific
competition as the unique mechanism. Generic added-row, position, or recency
sensitivity remains a live alternative, and strict-matcher-unmatched E rows
cannot be equated with unsupported output.

No training, inference policy, architecture, or final-set claim is promoted.

## Owning evidence

| Product | Path or receipt |
| --- | --- |
| Analysis | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification/20260804T060015Z/geometry-analysis/` |
| Analysis receipt self-seal | `be7eb24db57287d169fa6ec6e1ffcba34a1f933fb9654378ce27850fb3df95ed` |
| Visuals | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification/20260804T060015Z/geometry-visuals/` |
| Visual manifest self-seal | `bbc4af48d4c26d6c94a3275f64ddf514e1101ec5854a2cab529345d8c7d17bee` |
| Frozen rows | `26` C/E primary, `26` C/F sensitivity, `12` target/displacer pairs |
| Images | frozen human-refined twelve-image panel |
| New model execution | none |

The earlier `20260804T055623Z` artifact is superseded plumbing evidence: its
numeric bytes agree, but its declared visual root was unusable. It is not the
owning artifact.

## Primary material-negative rows

| C owner | C -> E description | E match | Geometry | Relative coordinate delta |
| --- | --- | --- | --- | ---: |
| `gt:13348:7` | truck -> person | unmatched | clearly separated | `-1.995` |
| `gt:13923:14` | chair -> chair | unmatched | overlap/containment | `-9.019` |
| `gt:14038:19` | book -> book | unmatched | overlap/containment | `-1.447` |
| `gt:1584:8` | person -> person | unmatched | overlap/containment | `-4.039` |
| `gt:16228:11` | person -> umbrella | unmatched | clearly separated | `-2.301` |
| `gt:16228:38` | person -> person | unmatched | overlap/containment | `-2.928` |
| `gt:4134:27` | dining table -> tie | matched `gt:4134:28` | clearly separated | `-2.940` |
| `gt:4134:29` | wine glass -> wine glass | unmatched | clearly separated | `-2.624` |
| `gt:6040:13` | person -> person | matched `gt:6040:14` | overlap/containment | `-6.570` |

The three largest negative magnitudes are overlap cases. Across all 26 rows,
IoU has Spearman association `-0.532` with relative coordinate delta; every
leave-one-image-out coefficient retains the negative sign. Center distance
has association `+0.370`, again with stable leave-one-image-out sign. These
are descriptive associations, not causal estimates.

The optional P+E+C->F sensitivity has `7` material-negative rows and cannot
vote. It does not overturn the primary route.

## Posthoc visual-support adjudication

This single-rater, original-resolution table was added after the frozen
numeric route and cannot change its denominator or threshold. `Unmatched`
means only that the row failed the current strict one-to-one GT matcher. It
does not mean hallucinated. The verdicts below record visible support, not
annotation truth, and therefore separate support from whether a missing
annotation, owner-identity mismatch, or geometry error caused the strict
failure.

| Owner | Unmatched E | Visual-support verdict | Bounded reading | Executed media SHA-256 |
| --- | --- | --- | --- | --- |
| `gt:13348:7` | person `[559,632,560,683]` | `supported_geometry_failure` | A visible ground-crew person is present at the location, but the one-bin width is severe geometry collapse; missing annotation versus failed geometry remains unresolved. | `dc783dc72da313e57014d274a97dfb3e9a311443c2fc4101f5dc93856fe3b874` |
| `gt:13923:14` | chair `[717,572,774,802]` | `supported_geometry_failure` | Local/extent rendition of a visible chair; may be the same physical owner under poor geometry rather than a new unlabeled chair. | `05e9188d7f9de9d6763879960e9c8217c8521efa5c3bab2269429b54f2359944` |
| `gt:14038:19` | book `[797,552,849,557]` | `supported_geometry_failure` | A visible book reference with a `52 x 5` collapsed extent; owner identity remains ambiguous. | `0e403e308441bb275578f177bc0424401ea1a200aa463ab9698f6dde58c68575` |
| `gt:1584:8` | person `[486,552,520,599]` | `supported_geometry_failure` | Local/extent rendition of a visible person; strict mismatch alone cannot decide missing annotation versus geometry. | `0e0b9ab0039b6165e6e6e29fc713ae34590a7a421a26c6752bba812f6a9ee872` |
| `gt:16228:11` | umbrella `[458,388,480,414]` | `plausible_support_extent_uncertain` | A plausible canopy or object part is visible, but the umbrella label and physical extent are not reliable enough for a stronger claim. | `3d166106d64142b8bb5f3cc9128bb7799d040f67e7fced9e8ad94b3bdc10e96a` |
| `gt:16228:38` | person `[72,461,105,585]` | `supported_geometry_failure` | Local/extent rendition of a visible person; annotation relation remains unresolved. | `3d166106d64142b8bb5f3cc9128bb7799d040f67e7fced9e8ad94b3bdc10e96a` |
| `gt:4134:27` | tie `[630,452,677,852]` | `matcher_verified` | Strict-matches a different real GT owner, `gt:4134:28`. | `9aa1d2e4148baa817bf7761b293d5cdc116caddf5527d59fda2387203aca6eb8` |
| `gt:4134:29` | wine glass `[0,473,17,507]` | `plausible_support_extent_uncertain` | A plausible partial edge object is visible, but the crop does not support a confident physical-owner or annotation judgment. | `9aa1d2e4148baa817bf7761b293d5cdc116caddf5527d59fda2387203aca6eb8` |
| `gt:6040:13` | person `[591,576,618,641]` | `matcher_verified` | Strict-matches a different real GT person, `gt:6040:14`. | `ac89a391fbb907877c9e8c6dbca3279bf4a841648b3303e0eb28b11ff86c8c47` |

No material-negative E is a clear hallucination: five unmatched rows are
supported renditions of visible objects with degraded geometry, two are
plausible but uncertain, and two are matcher-verified. This strengthens the
need for unknown-neutral treatment; it does not prove that all seven unmatched
rows are missing annotations. A drop in the exact corrupted coordinate string
also does not establish that eventual owner coverage is harmed or helped,
because this unit does not observe where the released probability mass moves.

## Greedy displacement compatibility check

Only `4/12` target/displacer pairs overlap or contain one another; `8/12` have
zero IoU. Visual review shows that zero IoU is not the same as distant
scattering. The bounded reading is:

> non-duplicate, predominantly near-neighbor same-category reprioritization:
> displacers rarely overlap targets, but usually sit close in image space and
> sorted order.

Nine of twelve pairs have absolute sorted-rank gap at most three. This panel
is descriptive only and cannot change the C/E decision.

## What changed in the mechanism map

- Identity/extent ambiguity remains important and carries the largest
  coordinate-delta magnitudes.
- It is not sufficient: four material rows are clearly separated, including
  one strict-matched real tie.
- Overlap does not imply duplicate: the overlapping E for `gt:6040:13`
  strict-matches a different physical person.
- A generic insertion/position/recency effect remains observationally joined
  with owner-specific competition because E adjacency was constructed, not
  discovered.
- Unmatched E rows remain unknown-neutral; human omissions and plausible
  local physical support are live.

## Artifact errata

The sealed `geometry-report.md` has two wording defects that do not affect
numeric products:

1. its materiality glossary says "the same four coordinate tokens"; the unit,
   analyzer receipt, and following glossary line correctly state that the
   crossing and benign deltas score their respective downstream rows, each
   over its own four coordinate tokens; and
2. its pair label `distributed_same_category_reprioritization` overstates
   spatial distance. The qualified wording in this result owns the
   interpretation.

The immutable artifact was not rewritten.

## Next discriminator

Exactly one new owning unit may propose a matched-length added-row control on
the frozen twelve-image panel. It should keep the same C/E boundary and exact E
coordinate tokens, but replace inserted C with a real same-image GT row that
is already covered, differs in category from C, is geometrically distant from
C and E, and is token-length matched where possible. The control must record
all length deltas and remain unknown-neutral for ambiguous E rows.

- If the neutral insertion reproduces materiality for most C-material owners,
  generic insertion/position sensitivity should close the owner-specific
  competition reading at this boundary.
- If neutral insertion is nonmaterial where C remains material for at least
  three owners across at least two images, C-specific interference survives.
- Otherwise stop inconclusive.

This result only makes that control eligible; it does not authorize it. Image
`2299` remains prospective-only for a later versioned thirteen-image
replication/prevalence unit and is not inserted into this control.
