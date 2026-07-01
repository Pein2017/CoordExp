# FN Book Exact Context-Ring Band Patch

Date: 2026-06-12

## Scope

This probe decomposes the prior `context_ring` intervention for the hard book
false negative into an exact three-band partition:

```text
image_id = 139
gt_idx = 17
desc = book
target y2 = 826
prompt tier = desc_x1_y1_x2
prefixes = 0, all
```

The previous whole-ring patch showed strong causal leverage but unstable
coordinate behavior: rank improved sharply, while top1 could jump to the
over-extended `999` basin. This run asks which part of the ring carries that
leverage.

## Discarded First Attempt

An earlier row-band artifact was generated under:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_row_bands
```

Do not use it for durable interpretation. It did not preserve the original
context-ring partition and target exclusion: its band boxes extended outside
the original 28-token context ring and included tokens that were not part of
the intended intervention support.

## Corrected Artifacts

Corrected candidate regions:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_exact_bands/candidate_regions
```

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_exact_bands/no_aligner_parent_ckpt3668
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_exact_bands/aux_latest_ckpt32
```

Shared reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_exact_bands/book17_exact_context_ring_band_patch_reduction.json
```

Counts per checkpoint:

```text
baseline_row_count = 2
membership_row_count = 20
patch_row_count = 48
```

Patch layers:

```text
13,14,16,17,24,25,26,27
```

## Exact Band Partition

The original context ring is:

```text
[802,803,804,805,806,841,842,843,844,845,880,881,882,884,
 919,920,921,923,958,959,960,961,962,997,998,999,1000,1001]
```

The corrected bands preserve target exclusion and union exactly to that ring:

| band | token count | tokens |
| --- | ---: | --- |
| `context_ring_upper_band` | 10 | `[802,803,804,805,806,841,842,843,844,845]` |
| `context_ring_target_adjacent_band` | 8 | `[880,881,882,884,919,920,921,923]` |
| `context_ring_lower_band` | 10 | `[958,959,960,961,962,997,998,999,1000,1001]` |

Band boxes used to build those exact supports:

| band | source box |
| --- | --- |
| `context_ring_upper_band` | `[880,653,999,730]` |
| `context_ring_target_adjacent_band` | `[880,730,999,807]` |
| `context_ring_lower_band` | `[880,807,999,890]` |

All three preserve the original target exclusion box.

## Baselines

| checkpoint | prefix | target rank | target prob | top1 | top1 distance | top bins |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| no-aligner parent | `0` | 131 | 0.000696 | 737 | 89 | `[737,736,739,743,741,740,730,734]` |
| no-aligner parent | `all` | 130 | 0.000650 | 739 | 87 | `[739,734,737,743,736,730,740,741]` |
| aux checkpoint | `0` | 161 | 0.000219 | 730 | 96 | `[730,731,722,734,723,718,724,725]` |
| aux checkpoint | `all` | 155 | 0.000278 | 734 | 92 | `[734,731,730,736,723,722,725,726]` |

## Band Results

Best rank movements by band:

| checkpoint | prefix | band | best layer | baseline rank | best rank | recovery | prob delta | patched top1 | top1 distance |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-aligner parent | `all` | `upper` | 13 | 130 | 62 | 68 | +0.005240 | 743 | 83 |
| no-aligner parent | `all` | `lower` | 13 | 130 | 103 | 27 | +0.000858 | 743 | 83 |
| no-aligner parent | `all` | `target-adjacent` | 13 | 130 | 118 | 12 | +0.000203 | 739 | 87 |
| no-aligner parent | `0` | `upper` | 16 | 131 | 96 | 35 | +0.002987 | 751 | 75 |
| no-aligner parent | `0` | `lower` | 13 | 131 | 123 | 8 | +0.000054 | 743 | 83 |
| no-aligner parent | `0` | `target-adjacent` | 24 | 131 | 130 | 1 | -0.000002 | 737 | 89 |
| aux checkpoint | `all` | `upper` | 13 | 155 | 78 | 77 | +0.004883 | 778 | 48 |
| aux checkpoint | `all` | `lower` | 13 | 155 | 138 | 17 | +0.000334 | 734 | 92 |
| aux checkpoint | `all` | `target-adjacent` | 13 | 155 | 140 | 15 | +0.000271 | 734 | 92 |
| aux checkpoint | `0` | `upper` | 16 | 161 | 117 | 44 | +0.002272 | 769 | 57 |
| aux checkpoint | `0` | `lower` | 13 | 161 | 148 | 13 | +0.000098 | 734 | 92 |
| aux checkpoint | `0` | `target-adjacent` | 13 | 161 | 154 | 7 | +0.000054 | 734 | 92 |

The upper band dominates in both checkpoints and both prefix regimes:

```text
aux prefix 0:   upper recovery 44
aux prefix all: upper recovery 77
no-aligner prefix 0:   upper recovery 35
no-aligner prefix all: upper recovery 68
```

No band is a clean rescue. The best patched top1 is aux `all`, upper band,
layer 13:

```text
baseline top1 = 734
patched top1 = 778
target y2 = 826
remaining distance = 48 bins
```

## Mechanism Read

The broad context-ring effect is not driven by lower/background tokens alone.
The dominant causal subregion is the upper context band above the target-adjacent
row, even though it does not include the direct target-overlap tokens.

This sharpens the false-negative mechanism:

1. The hard book y2 failure is not simple visual blindness. Perturbing a
   nearby visual neighborhood changes the y2 distribution substantially.
2. The direct target-adjacent band is weak. Direct edge tokens and immediate
   overlap are not sufficient handles for rescue.
3. The lower band has some leverage, but much less than the upper band. This
   argues against the prior whole-ring `999` jump being only a lower/background
   artifact.
4. Aux is more responsive to the upper band than no-aligner, reaching a closer
   top1 of `778`, but the coordinate basin still does not stabilize at `826`.

The current picture is broader visual-neighborhood control plus coordinate-basin
attraction. The model can be steered by visual context, but the y2 slot remains
pulled toward short or mid-extent coordinates instead of the correct lower edge.

## Next Hook

The promising path is to test whether the upper band is acting as a semantic
row-context anchor, a coordinate-height prior, or an indirect disambiguator for
stacked same-desc books. Two useful continuations:

1. Repeat exact-band decomposition on additional `hard_no_rescue` false
   negatives with overlapping same-desc visual tokens.
2. Patch upper-band residuals between prefix conditions or between checkpoints,
   not only zero them, to separate "remove misleading context" from "supply
   missing aligned context."

## Verification

Artifact verification checked:

```text
aux verified prefixes: ['0', 'all'] upper best: {'0': 44, 'all': 77}
no_aligner verified prefixes: ['0', 'all'] upper best: {'0': 35, 'all': 68}
VERIFIED exact band invariants, upper-band dominance, and no clean y2 rescue
```

The verification asserted:

- each checkpoint root has baseline `2`, membership `20`, and patch `48`;
- each checkpoint root reports `band_union_equals_context_ring = true`;
- each checkpoint root has the exact 28-token context ring listed above;
- each checkpoint root has the exact three band token lists listed above;
- for both checkpoints and both prefixes, `context_ring_upper_band` is the best
  band by maximum rank recovery;
- the best patched top1 still remains at least 48 bins from the gold `826`.
