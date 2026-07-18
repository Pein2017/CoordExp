---
title: Historical Random versus Geometry-Sorted Image 2299 Next-Row Screen Results
type: investigation
role: research-results
authority: executed_evidence
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Historical Random versus Geometry-Sorted Image 2299 Next-Row Screen Results

## Verdict

Under one identical dense image, prompt, three-row forced prefix, precision,
and sampling policy, the two historical adapters implement materially different
next-row behavior:

- the random-order adapter is almost insensitive to which treatment person was
  just appended and sends all 96 sampled continuations to the same person;
- the geometry-sorted adapter changes its generated first horizontal coordinate
  and physical person when the appended treatment person changes.

This rejects a confidence-rescaling-only explanation. It supports the narrower
interpretation that geometry-sorted supervision taught a prefix-conditioned
spatial transition habit, while random-order supervision left a comparatively
static image-conditioned candidate preference. It does **not** establish a
coverage ledger, an optimal traversal rule, or a general advantage from one
image and one training seed.

No architecture or training method is promoted by this unit.

## Executed contract

The frozen benchmark pair is:

- random-order checkpoint:
  `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668`;
- geometry-sorted checkpoint:
  `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668`.

Both adapters used the Qwen3 Vision-Language 2-billion-parameter base, pure
cross-entropy supervision, four epochs, effective batch size 128, Low-Rank
Adaptation rank 8 and alpha 32, and one training seed. Adapter SHA-256 checksums
were respectively:

```text
random: 3c2037907c476e6e6ebecd4b5a6439d6a32e75a560ac0d8de0f1994ebf6174f6
sorted: 7c2f880d63e2bdc2c114bf9b5edbea104852e2426fb4380ecaefb83a573aa95a
```

All model computation used full-model 32-bit floating point and eager
attention. Every arm verified that the active historical 1002-row coordinate
offset tensor exactly equalled the checkpoint tensor after hook reattachment:

```text
maximum absolute difference in 32-bit floating point = 0.0
```

The prompt hash was:

```text
6c03d11f37f6ebd2676d6eae2a1a3329534243b512bbfdaed0bacbd5c4d6cca2
```

The image-2299 authority contained 38 `person` rows and 8 `tie` rows. Every
stage-one arm scored all 46 rows. Stage Two used the identical seeds `0..23`,
temperature `0.4`, top-p `0.95`, repetition penalty `1.0`, and a nine-token
safety horizon with structural stopping after one complete legacy row.

## Stage One: exact canonical-row likelihood

The prefix always contained person-only ranks 0 and 1 followed by one treatment
owner. Owner 2 is the canonical geometry-sorted state. Owners 3, 4, and 14 skip
earlier people and are challenge states, not exchangeable replications.

| Treatment owner | State | Random / sorted rank of the just-emitted owner | Random / sorted candidate-restricted exact-row entropy | Cross-adapter Pearson correlation |
|---:|---|---:|---:|---:|
| 2 | canonical `[0, 1, 2]` | `40 / 7` | `2.312 / 1.325` | `0.215` |
| 3 | skipped-prefix challenge | `23 / 6` | `2.291 / 1.461` | `0.219` |
| 4 | skipped-prefix challenge | `32 / 4` | `2.289 / 1.177` | `0.168` |
| 14 | deep skipped-prefix challenge | `36 / 12` | `2.286 / 1.897` | `0.314` |

Top-1 canonical-row agreement was zero in all four states. The affine score
fits explained only about `2.8%` to `9.9%` of the cross-adapter variance. The
adapters therefore did not differ by a simple global confidence scale.

The random adapter's complete-row distributions were very stable across the
four prefixes: pairwise Pearson correlations were `0.943` to `0.981`, and four
or five of its top five candidates were shared by every pair of states. It
placed the same person-only rank 33 first in every state.

The geometry-sorted adapter was state-dependent. Its top candidate moved from
person-only rank 6 after owner 2 or 3, to rank 3 after owner 4, and to a `tie`
row after owner 14. Its owner-14 distribution had no top-five candidate in
common with its owner-2, owner-3, or owner-4 distributions.

### Why complete-row likelihood was not enough

All person candidates share the same description token. Across the 46 rows,
the sorted-versus-random mean absolute log-probability difference was roughly:

| Component | Mean absolute difference across the four states |
|---|---:|
| Description | `0.05` to `0.26` |
| `x1`, the left boundary | `0.79` to `0.88` |
| `y1`, the top boundary after teacher-forced `x1` | `4.07` to `5.53` |
| `x2`, the right boundary | `0.37` to `0.40` |
| `y2`, the bottom boundary | `0.36` to `0.43` |

Most full-row disagreement therefore appeared at `y1` after a candidate's
`x1` had already been supplied by teacher forcing. A high canonical-row score
could reflect coordinate completion after a forced left boundary rather than
free next-object selection. This is why Stage Two was required.

The terminal token was not competitive at these forced states. The
object-row-start minus terminal-token margin remained `9.68` to `11.07` in all
eight arms. The immediate question was which row followed continuation, not
whether generation continued.

## Stage Two: one-row conditional sampling

All 192 generations produced exactly one syntactically valid `person` row:

```text
2 adapters x 4 treatment owners x 24 paired seeds = 192 rows
terminal = 0
malformed = 0
non-person description = 0
```

`Strict match` requires exact category and bounding-box Intersection over Union
of at least `0.5`. `Nearest person` reports the highest-overlap relabeled person
even when geometry misses that threshold; it must not be interpreted as a
ground-truth match.

| Adapter | Treatment owner | Strict physical-person result | Nearest-person attribution |
|---|---:|---|---|
| random | 2 | person 25: `24/24` | person 25: `24/24` |
| random | 3 | person 25: `24/24` | person 25: `24/24` |
| random | 4 | person 25: `24/24` | person 25: `24/24` |
| random | 14 | person 25: `24/24` | person 25: `24/24` |
| sorted | 2 | person 6: `14`; person 3: `3`; person 4: `1`; unmatched: `6` | person 6: `20`; person 3: `3`; person 4: `1` |
| sorted | 3 | person 4: `6`; person 2: `2`; unmatched: `16` | person 2: `17`; person 4: `7` |
| sorted | 4 | person 3: `24/24` | person 3: `24/24` |
| sorted | 14 | person 10: `12`; person 18: `11`; person 25: `1` | identical to strict match |

The 22 strict unmatched rows were not treated as hallucinations. Every one was
category `person` and spatially closest to a relabeled person. For sorted owner
2, all six were narrow or shifted variants nearest to person 6, with best
Intersection over Union from `0.314` to `0.499`. For sorted owner 3, most were
nearest to person 2, with some between persons 2 and 4. They are geometry or
instance-binding failures under this automatic review, not evidence that the
physical entity was absent.

The generated first horizontal coordinate exposed the policy difference before
later teacher-forced completion:

- random generated `x1 = 0` in all 96 samples and always reached person 25;
- sorted owner 2 generated a band leading mostly to person 6;
- sorted owner 3 generated `x1` near `660..684`, leading to persons 2 or 4;
- sorted owner 4 generated `x1` near `290..306`, leading to person 3;
- sorted owner 14 was bimodal, with twelve `x1 = 0` rows and twelve near
  `814..841`, leading to persons 18 and 10 respectively, except one person-25
  sample.

Only one of 96 paired random-versus-sorted seeds reached the same strict
physical person. No sampled row matched its treatment owner or either shared
parent. This is only an observed absence, not evidence of exact-self
suppression: the random adapter always preferred person 25, and the tested
treatment owners did not include that preferred successor. A valid suppression
test must prefix the random adapter with person 25 and the sorted adapter with
each state-specific dominant successor.

Stage One's highest random canonical row was person 33, while Stage Two always
generated person 25. Stage One also assigned the sorted owner-14 candidate set
about 45% restricted probability to `tie` rows, while Stage Two generated no
`tie`. These two mismatches directly demonstrate that teacher-forced
complete-row ranking is not a substitute for the model's autoregressive path.

## Interpretation from first principles

Let `B_t` be the geometry encoded by the latest emitted row and `Z_(t+1)` be
the next physical object. In geometry-sorted training, `B_t` carries
information about where a valid successor is likely to be. A decoder can learn
a compact conditional rule:

```text
next-object distribution = function(image, emitted prefix geometry)
```

Under random permutation, the next target is deliberately made almost
independent of the latest row geometry. Pure token cross-entropy then gives the
model no consistent transition rule to learn. On this image, the random adapter
falls back to a stable image-wide preference and repeatedly chooses the same
person regardless of the latest emitted row.

This explains why random serialization is not sufficient evidence for
order-independent set enumeration. Removing a fixed order also removes a cheap
state signal; it does not create a covered-set mechanism. Conversely, the
sorted adapter's useful prefix-conditioned routing is still not a coverage
ledger: it can backtrack, it depends strongly on the latest box, and this unit
does not test memory beyond one appended treatment row.

The result is consistent with the historical natural image-2299 rollout: the
random adapter entered an invalid repetition burst, whereas the sorted adapter
produced 20 parseable rows. That historical rollout is supporting context, not
part of the controlled 32-bit floating-point execution in this unit.

## Hypothesis update

| Hypothesis | Outcome | Reason |
|---|---|---|
| The adapters differ only in confidence scale. | rejected for this state | Low cross-adapter correlation, zero top-1 agreement, and different sampled owners. |
| Random ordering naturally teaches order-independent coverage. | rejected for this state | All four prefixes led to person 25 with the same `x1`. |
| Geometry sorting teaches a prefix-conditioned spatial transition. | supported for this state | Changing only the latest treatment row changed generated `x1` and physical owner. |
| Geometry sorting establishes a general covered-set ledger. | unresolved | No sampled row matched the selected treatment owner, but the tested owners were not each adapter's preferred successor and no multi-row state was measured. |
| Early stopping explains the adapter difference here. | rejected for this state | Both adapters strongly preferred row start, and Stage Two produced zero terminal outcomes. |
| The difference is only later coordinate completion. | rejected as a complete explanation | The adapters already generated different `x1` values and physical owners. Later completion still contributes strongly. |

### Subsequent dominant-owner correction

The later [Person 25 Dominant-Owner Commit and Persistence Closeout](../2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/results.md)
fills the missing treatment identified above. A person-25 row does redirect the
random adapter toward overlapping person 18, both with raw donor rows and
inside natural three-row continuations. Therefore the row
"random ordering naturally teaches order-independent coverage" should be read
as **not established by this historical screen**, not as a valid proof that the
random adapter has no commit-like transition. The closeout still does not find
a reliable covered-set ledger: person 18 repeats after earlier commitment, and
the greedy path is a merged box resolved mainly at `y2`.

## Stopping decision and next discriminator

This unit stops here. It has established a real training signature without
requiring a new architecture.

The smallest next experiment is an `x1`-conditioned completion crossover:

1. sample or freeze the same prefix, description, and generated `x1`;
2. give that identical partial row to both adapters;
3. compare the resulting `y1`, `x2`, and `y2` distributions and final physical
   owner;
4. repeat on four to eight densely relabeled images before drawing a general
   conclusion.

This separates how much supervision changed next-region selection from how
much it changed box completion after a region was partly selected. A training
screen is not justified until that split is known.

## Artifacts

Artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen
```

Primary deterministic summary:

```text
summary/summary.json
SHA-256: 3dfac7d5ca0fb553b1d3fed939e5ba64ccb1a5a61525d6d60a75d2a81c5f2415
```

Raw evidence is retained under `arms/` and `sample-arms/`. The executable is:

```text
scripts/research/run_historical_random_sorted_image2299_screen.py
```

Model-free verification:

```text
15 passed
```

## Limitations

- one image;
- one historical training seed per ordering policy;
- forced, human-relabel rows rather than naturally reached historical states;
- only owner 2 is canonical under geometry-sorted training;
- automatic physical-owner assignment depends on exact category and geometry;
- no multi-row memory horizon was tested;
- no causal claim about the ordering policy across training randomness;
- no claim about general detection quality, hallucination, or Common Objects in
  Context annotation completeness.
