---
title: Local Branch Causality and Downstream Unique-Object Value Results
description: Full-model float32 discovery found no admissible same-prefix covered-duplicate versus sampled-uncovered pair, while exposing stronger path-mediated access to sampled-only owners.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
unit_id: 2026-07-19-local-branch-causality-and-downstream-coverage-value
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_insufficient_cohort
updated: 2026-07-19
---

# Local Branch Causality and Downstream Unique-Object Value Results

## Verdict

The preregistered primary phenotype was not available at useful frequency.
Across the frozen eight-image panel, zero exact natural prefixes contained both:

1. a native greedy complete row owned by a verified already covered object; and
2. a naturally emitted sampled complete row owned by a verified uncovered
   object of the same category and tokenized length.

The minimum required cohort was four images. The frozen result is therefore
`insufficient_cohort`, the planned token and complete-row causal arms did not
run, and no local branch-ranking training proposal or architecture change is
promoted.

This is not evidence that local token steering is impossible. It shows that
the particular covered-duplicate-versus-uncovered same-prefix phenotype is too
rare, or too difficult to resolve safely, to carry the proposed experiment.

## Numerical and execution evidence

The primary model was the description-first, geometry-sorted,
pure-cross-entropy-plus-token-type-gate Weight-Decomposed Low-Rank Adaptation
checkpoint at step `4,887`. All model parameters were loaded in 32-bit floating
point. Hugging Face inference used physical batch size one and repetition
penalty `1.0`.

Before discovery, image `2299` passed an executable no-op replay smoke. Exact
raw token identifiers were equal for the native branch row, the successor after
appending the native branch token, and the successor after appending the native
complete row.

| Artifact | Secure Hash Algorithm 256-bit digest |
|---|---|
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-local-branch-causality-and-downstream-coverage-value/noop-smoke-image2299-fp32-v2/smoke.json` | `27dcdf3ad2cd5c540086393b420d8c66bbda3ee557031f52c48231ddd47a316f` |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-local-branch-causality-and-downstream-coverage-value/strict-primary-case-freeze-v1/admitted-cases.json` | `7dd9984b4828a5c9ad0605d7bf91aa792ae41b63d1bb6039363de77fbbe4b4c4` |

Each image then completed one greedy and eight sampled root trajectories with
fixed sampled seeds `11` through `18`. Every exact deduplicated natural prefix
was replayed greedily. The eight immutable discovery receipts are:

| Image | Discovery artifact digest |
|---|---|
| `2299` | `fe5f68da4c3106fa76c0d19a1001bd5b2a5f39829497aece8c4b03e156628abc` |
| `7816` | `3a33b56d0ff6f1a25f440926e303a9ec83f0b20f1a5fd8c2296da54702adcf64` |
| `9400` | `e237c369acf204d72e50ce02b0bad0f0ff46feb5756946cbcf6ab3703b283ec0` |
| `9590` | `4d79c5799a690f3c89137359b3dbd7bfd955dba44311bfc254316892cf298917` |
| `12576` | `46689131b7f81ce15dad75916c0ed908cff67c5f0a0b363bb542473f40fde3b9` |
| `18380` | `ad67f393086e5ed67177eaf95a99ee82c58a0c25472d4b809f8dde7ffbe72f26` |
| `19109` | `e19509b3e88e7f6afb5a0d8e8fa8f3fb96e175c0d6eb728c3f861f8c76ed8c59` |
| `19432` | `df1fc17a22b36d831ca2d158e96ebfa357339dd8486418f3d6176eb7bf13d645` |

## Frozen admission result

| Image | Exact prefixes replayed | Admissible primary pairs |
|---|---:|---:|
| `2299` | `64` | `0` |
| `7816` | `64` | `0` |
| `9400` | `64` | `0` |
| `9590` | `62` | `0` |
| `12576` | `64` | `0` |
| `18380` | `64` | `0` |
| `19109` | `64` | `0` |
| `19432` | `61` | `0` |

The two clean native covered-duplicate prefixes both occurred on image `7816`.
One had no natural sampled next action. The other had a sampled `person` row of
the same nine-token length, but its physical owner was unresolved. Its box was
closest to the already covered person `205108` with Intersection over Union
`0.424`, and only `0.052` with uncovered person `211764`. Even an explicit
review override could produce at most one admitted image, still below the
four-image gate.

The aggregate refusal counts were:

| Reason | Count |
|---|---:|
| Earlier prefix ownership became unresolved | `267` |
| Native greedy row was not a verified covered duplicate | `211` |
| Sampled row ownership was unmatched or ambiguous | `52` |
| Sampled owner was already covered | `4` |

The `211` records are correlated prefix-action observations, not independent
images or unique objects. In the frozen JSON they are grouped under the label
`verified_benign_uncovered_owner`; operationally this means only that the
sampled action has an uncovered verified owner while the native row is not a
verified duplicate. It does not label the native row itself as benign.

## Stronger observation outside the failed primary phenotype

Within the fixed eight-row horizon, sampled trajectories contained fourteen
strictly matched physical owners that the paired root-greedy trajectories did
not contain. They occurred on six of eight images.

At the exact sampled prefix immediately before those owners were emitted,
native greedy replay produced:

- the same physical owner in `6 / 14` cases;
- a different verified uncovered owner in `4 / 14` cases; and
- an unresolved row in `4 / 14` cases;
- no covered duplicate, terminal row, or malformed row in any case.

This changes the leading question. Many sampled-only owners are not locally
difficult after the sampled trajectory has reached the corresponding prefix.
The unresolved problem is how earlier sampled row tokens produce a state from
which those owners become greedily accessible.

Physical owner sequence alone is not a sufficient state description. In
images `12576`, `18380`, and `19432`, sampled and exact-prefix greedy replay can
follow the same physical-owner sequence toward the target while their raw row
tokens already differ earlier. Small description or coordinate differences may
therefore carry later route state even when the coarse owner order is unchanged.

## Important horizon limitation

Every root-greedy trajectory generated all eight allowed rows. None reached a
natural terminal within the discovery budget. Consequently, `sampled-only`
here means only absent from the first eight greedy rows. It does not prove that
the owner is absent from a longer greedy trajectory.

Any follow-up must first extend root-greedy decoding under the same numerical
and decoding contract. Owners that appear later under greedy decoding are
route accelerations or reorderings, not unique sampled support. Only owners
still absent at a declared longer cap or natural terminal can enter a stronger
path-accessibility claim.

## Hypothesis update

### Not supported by this cohort

The planned common failure mode—native greedy repeats a covered object while
sampling chooses an uncovered sibling at the same exact state—did not recur
often enough to study. The unit therefore provides no evidence for training an
earliest-token preference at that phenotype.

### Bounded support

Sampling can alter earlier row tokens and reach later prefix states where an
owner absent from the first eight root-greedy rows is already easy for greedy
decoding. This supports path-mediated object accessibility as a more useful
next hypothesis than a purely local sampled-token rescue.

### Still unresolved

- whether the apparent sampled-only owners remain absent from longer greedy
  rollouts;
- which earlier complete row is sufficient to make a chosen owner accessible;
- whether one local branch token is sufficient, or several row-level changes
  must accumulate;
- whether such a route improves total safe unique-object coverage rather than
  only changing order; and
- whether a successful treatment can be expressed as a loss on the native
  prefix state or requires an additional state carrier.

## Decision and next discriminator

Do not relax the failed admission rule, add seeds, expand the image pool, run
the planned intervention arms, or start training from this unit.

The smallest evidence-changing successor is a separate frozen unit with two
stages:

1. extend greedy root trajectories under the same full-float32, batch-one,
   repetition-penalty-`1.0` contract to distinguish late greedy discovery from
   genuinely sampling-dependent access; and
2. for surviving targets, walk backward through the exact sampled trajectory
   and compare from the same parent prefix:

```text
append the natural sampled complete row, then release greedy decoding
versus
append the native greedy complete row, then release greedy decoding
```

Run complete-row comparisons before token-level interventions. If one sampled
row safely makes the target or a larger unique set greedily reachable, only
then split that row at its earliest differing raw token. If no single row is
sufficient but the accumulated sampled prefix is, the evidence favors a
distributed prefix-state problem rather than a local token-ranking treatment.

## Verification

The experiment-local runner and freezer are covered by `24` deterministic
tests. Python bytecode compilation and scoped diff checks pass. An independent
read-only audit reproduced the zero-case result and confirmed that no pair was
removed only by the same-category or equal-row-length filters.

## Limitations

- The panel is a selected eight-image mechanism study, not a prevalence
  estimate.
- Positive-only owner ledgers prevent unsupported-output claims from annotation
  absence.
- Dense-scene geometry ambiguity invalidates many later coverage states.
- The eight-row cap prevents a final claim about greedy support.
- The follow-up path hypothesis is motivated by discovery evidence but has not
  yet received a causal complete-row intervention.
