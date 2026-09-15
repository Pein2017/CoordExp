# Current-Policy Annotated-Owner Set Signal Census Result

## Decision

- Mechanical status: **`MECHANICALLY_VALID`**.
- Scientific disposition: **`GO_MINIMAL_RLOO_VERTICAL`**.

Frozen C supplies substantially more current-policy annotated-owner set
variation than the registered minimum: `109 / 248` images have a fully clean
K=4 group with at least one strict IoU50 owner-set inclusion, versus the gate
of eight.  This licenses a separately contracted, one-update static
leave-one-out policy-gradient vertical.  It is signal-supply evidence only;
no optimizer step or model-quality result occurred here.

Authoritative analysis:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-current-policy-owner-set-signal-census/analysis-v3.json`

SHA-256:
`0569bdc818d69ddb221d18b57ae6ac22d7e2cb4938e3ed53e73ddbc2fd65496d`.

Lead acceptance receipt:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-current-policy-owner-set-signal-census/acceptance-v1.json`

SHA-256:
`25933a65740c7a78e05285ef17abd7f0965343c9481b8377801e337d40e30a5b`.

The acceptance replay verifies the 248-image/1,798-owner population, all eight
rollout artifact hashes, strict-inclusion pairs, centered RLOO advantages,
safety counts, and the annotation-quality route counterfactual.

## Executed specimen

The census used frozen C as universal base plus its unmerged shared language
DoRA and registered embedding delta.  It sampled four natural completions for
each of 248 train images with seeds `2026090201..2026090204`, temperature
`1.0`, top-p `1.0`, repetition penalty `1.0`, and a 3,084-token cap.  Eight
independent one-GPU shards produced `992` trajectories.

The raw acquisition was launched at `d0ed592d7`.  Before any full-shard metric
was inspected, the K4 cleanliness and reward definitions were corrected as
recorded in the unit.  `analysis-v2` then failed before metric production on a
zero-padded C-anchor image-ID join.  Commit `5ff5b99e2` fixes only that key
normalization; `analysis-v3` reuses the unchanged raw shards.  There is no
`analysis-v1` scientific result.

For image `i` and rollout `k`, reward is
`r_ik = |S_ik| / |O_i|`, where `S_ik` is the category-consistent global
one-to-one annotated-owner set and `O_i` is the annotated-owner set.  The
leave-one-out advantage is
`A_ik = r_ik - mean_(j != k) r_ij`.  A dominance-bearing group requires all
four rollouts to be clean.

## Owner-set signal

| threshold | complete clean K4 | nonzero count range | strict-inclusion images | equal-count identity diversity | positive union-only gap | sampled set weakly expands C |
|---|---:|---:|---:|---:|---:|---:|
| IoU50 | 218 | 117 | **109** | 48 | 39 | 44 |
| IoU60 | 218 | 123 | 115 | 46 | 43 | 41 |
| IoU80 | 218 | 133 | 131 | 53 | 47 | 40 |

The IoU50 count-range histogram over the 218 fully clean groups is:

| range | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| images | 101 | 68 | 34 | 6 | 4 | 3 | 2 |

The signal therefore is not carried by one or two exceptional images, and it
does not disappear when the matching threshold is raised.  The 44 weak C
expansions show that sampled support sometimes contains all greedy owners plus
an additional annotated owner; they do **not** establish that a parameter
update will preserve that relation under greedy decode.

At IoU50, the 992 trajectories supply 4,776 matched annotated-owner slots and
3,328 valid unmatched rows.  The latter remain unknown and receive no negative
label.

## Density and annotation-quality counterfactual

| density | images | complete clean K4 | dominance-bearing |
|---|---:|---:|---:|
| low | 85 | 83 | 12 |
| medium | 83 | 74 | 48 |
| high | 80 | 61 | 49 |

High-density images provide more variation, but low- and medium-density images
also carry signal.  Nine high-density images meet the registered repeated
book/cup/fruit/vegetable review flag.  Only four are dominance-bearing.
Excluding all nine leaves `105` dominance-bearing images and the same
`GO_MINIMAL_RLOO_VERTICAL` decision, so conclusion-triggered visual review is
not required.  This preserves the user's distinction: sparse instances in
these categories remain reliable; only densely repeated instances receive the
quality sensitivity treatment.

## Safety and structural monitors

- `951 / 992` trajectories are fully clean.
- All `992` terminate naturally; there are zero caps, malformed rollouts, or
  invalid predictions.
- Forty-one natural-EOS trajectories contain 62 parser-dropped spans and are
  excluded from full-K4 dominance groups.
- There are zero exact row duplicates and zero exact trajectory-duplicate
  pairs; nine strict and 20 ambiguous physical-duplicate candidates are
  monitors only.
- There are 534 natural ordering inversions.  They are legal model behavior and
  do not affect eligibility or disposition.

## Interpretation

Observation: independent current-policy completions from the same image often
differ in both annotated-owner count and identity, with abundant strict set
dominance and clean image-normalized leave-one-out advantages.

Inference: a minimal within-image policy-gradient direction is empirically
identifiable at frozen C; no K8 support expansion or owner-identity-only
fallback is needed before testing one finite update.

Not established: that the estimator is implemented correctly, that an update
improves natural greedy coverage, that C owners are retained, or that behavior
transfers to image-disjoint data.  Selection of a small mechanism panel is
conditional evidence, not an unbiased COCO estimate.  The next unit must bind
the exact action-token span, DDP normalization, finite-step acceptance, cold
unmerged readback, train-248 and image-disjoint evaluation, and a stop on owner
exchange or global regression.  No PPO stack, value model, merged checkpoint,
or persistent external memory is licensed by this result.

## Claim boundary

This is a full-cohort, train-side, K=4 **signal census** from one frozen
checkpoint.  It establishes current-policy support for a one-update RLOO
mechanism test.  It is not learning, overfit success, held-out generalization,
partial-label robustness, architecture promotion, or production evidence.
