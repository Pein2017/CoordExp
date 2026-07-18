# Results: Human-Resolved Dense Branch Value and Calibration Screen

## Verdict

No sampled-only third-row branch produced a positive, safety-preserving
four-row value gain over the greedy third-row branch. The predeclared
own-prefix branch-preference training gate is therefore closed.

This is not a claim that the sampled branches are false. They are valid natural
person rows and they send the decoder into different, mostly non-repeating
routes. The result is narrower: on this exact manually relabeled dense state,
none of those routes is a better short-horizon target for deterministic greedy
enumeration.

## Execution

- Image: `2299`, manually relabeled with 38 people and 8 ties.
- Checkpoint: geometry-sorted Gaussian-supervision Weight-Decomposed Low-Rank
  Adaptation (`DoRA`) step `4,887`.
- Shared parent: two natural person rows, relabeled person ranks `1` and `0`.
- Natural third-row branches: person ranks `3`, `2`, `4`, and greedy rank `14`.
- Sampling: eight paired fresh seeds per branch, temperature `0.4`, top-p
  threshold `0.95`, repetition penalty `1.0`, physical batch size one.
- Horizon: four complete rows; all 32 calls produced four complete rows.
- Matching: same normalized category, Intersection over Union (`IoU`) at least
  `0.5`, and top-versus-second margin at least `0.05`; unmatched rows remain
  unresolved.

The immutable execution and analysis are at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/
image2299-four-branch-horizon4-paired-k8-v1/
```

## Primary Results

The primary count excludes the two parent people and the appended branch
person itself.

| Third-row branch | Role | Mean uniquely matched new people in next 4 rows | Difference from greedy, paired 95% interval | Mean safety failures |
|---|---|---:|---:|---:|
| person rank 14 | greedy | `3.500` | reference | `0.500` |
| person rank 3 | sampled-only | `2.875` | `-0.625` [`-1.375`, `0.125`] | `1.125` |
| person rank 2 | sampled-only | `2.250` | `-1.250` [`-2.000`, `-0.500`] | `1.750` |
| person rank 4 | sampled-only | `3.125` | `-0.375` [`-0.875`, `0.000`] | `0.875` |

The intervals use 20,000 deterministic paired bootstrap replicates over the
eight shared sampling seeds. Safety failure means an earlier-parent repeat,
branch repeat, within-suffix repeat, unresolved row, malformed row, or terminal
row. The greedy branch had four unresolved person boxes. The sampled rank-3
and rank-2 branches had nine and fourteen unresolved rows, respectively. The
sampled rank-4 branch had five unresolved rows, one malformed row, and one
repeat of its appended person.

There was no earlier-parent repeat in any arm and no within-suffix repeated
person. The central failure is therefore not a duplication burst or failure to
continue. It is that the alternative routes devote more rows to predictions
whose physical extent cannot be assigned reliably.

## Entity and Geometry Separation

Most unresolved rows after the sampled rank-3 and rank-2 branches say `tie`,
not `person`. Their best relabeled-tie IoU is usually far below `0.5`. A
crop-enlarged inspection of the affected top-row region confirms that real ties
are present there, but it does not establish that every predicted tie box owns
one particular tie. Several predictions are shifted, too narrow, or incomplete
relative to the manual tie boxes.

This matters for interpretation:

- the sampled branch can redirect semantic output toward a real object-rich
  part of the image;
- the resulting phrase-to-extent binding is not reliable enough to count as
  safe physical-object coverage; and
- official or relabeled mismatch must not be called hallucination without
  entity-level review.

The crop used for that review is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/
review/image2299-top-row-tie-region-4x.png
```

## Mechanism Update

The combined four-case evidence now supports the following bounded picture:

1. Sampling can reveal a real object mode hidden from greedy decoding at an
   exact state, as in image `12576`.
2. A natural object row changes the successor distribution and usually avoids
   immediate re-selection, as in image `2299`.
3. Different valid branches can enter substantially different spatial and
   semantic routes.
4. A sampled-only route is not automatically a higher-value route. Here the
   greedy route yields more reliably localized unique people over the next four
   rows, while sampled routes more often enter weakly localized tie output.

The evidence therefore rejects the simple training rule:

> if sampling finds a valid row that greedy misses, prefer that row over the
> greedy row at the same prefix.

That rule confuses mode diversity with downstream value. The treatment target
must include later unique physical-object coverage and phrase-to-geometry
quality, not only the first rescued row.

## Training Decision

The planned 256-image sampled-branch preference screen does not run. Its causal
precondition failed, so implementing it would spend compute teaching an
arbitrary route and could trade reliable people for poorly bound parts or
attributes.

This result does not close all loss-only approaches. It specifically redirects
the next calibration question toward branch-conditioned row binding:

- when an alternative route emits a plausible `tie` or another object phrase,
  is the correct physical extent already supported in the coordinate
  distribution but losing during autoregressive completion; or
- does that route contain only a semantic guess without a recoverable box?

A future training screen should be authorized only after that distinction
selects a concrete target, such as complete phrase-and-box consistency or a
short-horizon set-value objective. It should not reward sampled novelty alone.

## Verification

```text
conda run -n ms pytest -q \
  tests/analysis/test_analyze_image2299_horizon_branch_value.py
```

Result: `7 passed`.

The analyzer also passed Python bytecode compilation and scoped
`git diff --check`.

## Limitations

- One image and eight paired seeds do not estimate population prevalence.
- The four-row horizon can miss longer-term route convergence or recovery.
- Unresolved tie rows separate safe geometry from entity plausibility but are
  not fully human-adjudicated physical-owner labels.
- This experiment does not decide whether an explicit covered-set carrier is
  necessary.
