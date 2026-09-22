# Sampled Owner Inclusion Census

## Status

Completed on 2026-07-23.

This record separates two questions that must not be conflated:

1. **Full naturally closed trajectory support:** whether a physical owner
   appears anywhere in one of sixteen naturally closed sampled trajectories.
2. **Fixed sixteen-row support:** whether that owner appears within the first
   sixteen complete object rows of a sampled trajectory. This is the sampled
   side of the matched `Source@B16` comparison, where `B16` means a budget of
   sixteen complete object rows.

Neither census establishes sampled rescue relative to greedy decoding. That
claim requires the separately collected `Source@B16` owner set.

## Frozen Evidence

Source trajectory panel:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
trajectory-panel-2432-vllm/production-v2
```

The panel contains 2,432 images, sixteen sampled trajectories per image, and
38,912 trajectories in total. All trajectories ended with the natural image
end token; none ended because of the length limit.

Full-trajectory support artifact:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
sampled-owner-inclusion-v1
```

Fixed-sixteen-row support artifact:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
sampled-owner-inclusion-b16-v1
```

Both artifacts are immutable directories with source hashes, analyzer hashes,
matching-analyzer hashes, and hashes for every emitted content file. Their
recorded hashes were recomputed after publication and matched.

## Matching and Claim Boundary

For each trajectory, predictions are globally assigned one-to-one to annotated
physical owners with category agreement and Intersection over Union (IoU) at
least 0.50.

For an owner `o` in image `I`, define:

```text
q_lower(o | I) = unambiguous matched trajectory count / 16
q_upper(o | I) =
    (unambiguous matched trajectories
     union explicit ambiguity-candidate trajectories) / 16
```

The owner strata are:

- stable: `q_lower = 1`;
- occasional: `q_lower > 0` and `q_upper < 1`;
- unseen: `q_upper = 0`;
- uncertain: every remaining case.

Officially unmatched predictions remain unresolved. They are not counted as
owner support, negative examples, or hallucinations. Entity inclusion and box
geometry evidence remain separate.

## Main Counts

| Owner stratum | Full trajectory | First sixteen rows |
|---|---:|---:|
| Stable | 9,873 | 8,536 |
| Occasional | 8,312 | 8,248 |
| Unseen | 6,131 | 7,562 |
| Uncertain | 401 | 371 |
| Total | 24,717 | 24,717 |

The full-to-bounded transitions were:

| Full stratum to bounded stratum | Stable | Occasional | Unseen | Uncertain |
|---|---:|---:|---:|---:|
| Stable | 8,536 | 929 | 408 | 0 |
| Occasional | 0 | 7,305 | 996 | 11 |
| Unseen | 0 | 0 | 6,131 | 0 |
| Uncertain | 0 | 14 | 27 | 360 |

Therefore 2,385 annotated owners change classification when rows after sixteen
are removed. Full-trajectory support cannot substitute for fixed-budget
support.

Of the 38,912 sampled trajectories:

- 30,407 contained fewer than sixteen complete parsed rows;
- 1,072 contained exactly sixteen;
- 7,433 contained more than sixteen;
- the maximum was 83 complete rows;
- the mean was approximately 9.50 complete rows.

## Split Evidence

Full-trajectory owner counts were balanced across the frozen splits:

| Split | Owners | Occasional | Stable | Unseen | Uncertain |
|---|---:|---:|---:|---:|---:|
| Training candidate | 20,798 | 6,993 | 8,326 | 5,153 | 326 |
| Development | 2,616 | 890 | 1,033 | 638 | 55 |
| Held out | 1,303 | 429 | 514 | 340 | 20 |

For the fixed-sixteen-row view, 1,350 of the 2,048 training-candidate images
contain at least one occasional owner. This is only an upper bound on treatment
event supply, but it exceeds the planned 496-image broad bank by 854 images.

| Annotated object-count band | Training images with at least one bounded occasional owner |
|---|---:|
| 1 to 3 | 72 of 512 |
| 4 to 7 | 297 of 512 |
| 8 to 15 | 471 of 512 |
| 16 or more | 510 of 512 |

The candidate supply is strongly density-dependent. A balanced 496-image bank
cannot assume equal admission yield from all four bands; the separate Source
baseline and trust gates must determine whether the planned balance remains
feasible.

## Parser and Review Evidence

Across the full panel:

- 37,707 trajectories were parser-accepted;
- 1,205 were accepted with dropped spans;
- 5,559 predictions were mechanically classified as duplicates in the
  full-trajectory view;
- 135,016 predictions remained unresolved in the full-trajectory view;
- 4,371 duplicates and 105,547 unresolved predictions occurred within the
  first-sixteen-row view.

These counts are not hallucination rates. Incomplete Common Objects in Context
annotations and imperfect geometry prevent that interpretation. Dropped spans
also represent missing support evidence rather than negative owner evidence.

## Current Interpretation

1. Occasional owner discovery is a large, reproducible part of the model's
   sampled distribution, not a rare anecdote.
2. The phenomenon remains large under a common sixteen-row budget: 8,248 owners
   are occasional and 1,350 training images contain at least one such owner.
3. Dense images supply most candidate events. Sparse-image supply is the likely
   feasibility bottleneck for a strictly band-balanced bank.
4. The support-frequency distribution is not uniform. Occasional owners range
   from one successful trajectory out of sixteen to fifteen out of sixteen;
   event assembly must retain that frequency rather than treating every
   occasional owner as equivalent.
5. No training-feasibility or treatment-effect claim is yet authorized. The
   next required observation is the finite deterministic `Source@B16` owner
   set, followed by matched sampled-versus-Source route admission and the
   496-image gate.
