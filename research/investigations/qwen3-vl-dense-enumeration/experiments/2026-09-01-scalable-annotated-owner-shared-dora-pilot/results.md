---
title: Scalable Annotated-Owner Shared DoRA G0 Results
description: G0.4 supplied enough actual-prefix events, but the registered exact whole-bundle cross-image null was structurally infeasible and stopped the route before gradients or training.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed_to_registered_g0_stop
unit_id: 2026-09-01-scalable-annotated-owner-shared-dora-pilot
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_scientific_stop_before_gradient
conclusion_status: stopped_at_k1_whole_bundle_null_feasibility
updated: 2026-09-01
---

# Scalable Annotated-Owner Shared DoRA G0 Results

## Decision

G0.4 found ample actual-prefix signal: `440` annotated-owner positive event
contexts and `1,214` preservation event contexts, both above the registered
minimum of `64`. The following K1 preflight nevertheless found the exact
whole-bundle, different-image permutation null infeasible at the coarsest
one-bin level. Under the frozen v3 policy this is a registered
`SCIENTIFIC_STOP` with terminal code
`null_infeasible_at_1_bin_or_all_4_2_1_levels`.

The route therefore stops before deficits, gradients, G0.3, optimizer steps,
or G1 training. No checkpoint, merged export, dense payload, or adapter was
created. `HOLD_PRODUCTION` and no architecture promotion remain in force.

This is a negative result about the registered control construction, not about
whether shared DoRA can fit or generalize. Neither arm was trained, and no
image-disjoint model-quality comparison was executed.

## Immutable evidence

- Source checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
- Processed data:
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl`,
  with resize target `1024` and global maximum length `12000`.
- Scientific artifact root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-scalable-annotated-owner-shared-dora-pilot/g0/cross-image-v1/g0.4-source-signal-v3`.
- Receipt: `receipt.json`; receipt ID
  `e9884f53cfd6cf475fe39dd84e3126c9243b460f5885131f2961295d0a26326b`;
  file SHA-256
  `a6e9c4c29e337180232aecf0f52c25a62e6df276381b10b1b48b300212843490`.
- Frozen [v3 launch policy](g0-launch-policy-v3.json) SHA-256:
  `171976ddea0afad6fc6ea32565f2c013fc517ce26bbe4dfa401ea30960e70462`.
- Frozen [v3 research unit](unit-v3.md) SHA-256:
  `7d40b444f531dc677cb3ef11d1f792be0faf211681045978751ea02b8327b69b`.
- State-bank ID:
  `7ba5b8daaf93338e668ba8cfe485f03fabd08a961cde09fabf0dae0fcf3af624`;
  `1,654` records; records SHA-256
  `19058b880171b215adbbab23e40ef26d93abccc0b52008e684c1e97d934368bf`.

The receipt is the authoritative machine-readable result. Its execution
requests, Source trajectories, state-bank manifest, and state-bank records have
SHA-256 values
`10484cfdd7f86cebe4ce2d7a9fc74c443b0d1884076528932564a180b73c455e`,
`7bff43fea168ea637411c1c064e045933fe760b08ae34c4fe4ed58b585ec2099`,
`16804f5d6be805d0edee17a468081f25e89824a5ec5a04c80ee9128e610ac158`,
and `19058b880171b215adbbab23e40ef26d93abccc0b52008e684c1e97d934368bf`,
respectively.

## Mechanics acceptance

The corrected ordering semantics passed the six-file targeted suite with
`148 passed`, plus Ruff and compile checks. Historical v1/v2 receipts replayed
under their original four-coordinate ordering schema. The one-image
Image102420 runtime smoke then passed with status
`MECHANICS_SMOKE_VALIDATED`, receipt ID
`b4e3549f5248f3eed1722bd6c72fd4270c23168e06f5ee0600878427096cde33`,
and receipt SHA-256
`c43a323cd6d09dc35d3c14d0e60805bb2c01963c6ae2647377d7f76c88076509`.

That smoke admitted five Source-preservation rows and one positive row. In
particular, a naturally emitted preservation row earlier than an already
committed `(x1,y1)` anchor remained admissible. This directly checks the user
correction: natural decode need not be globally sorted.

## G0.4 denominators and resources

| Readout | Value |
| --- | ---: |
| requested train images | 256 |
| eligible images | 248 |
| dropped or malformed images | 8 |
| non-natural-stop images | 4 |
| parser-rejected images | 0 |
| images with invalid complete rows | 0 |
| quantized-degenerate owners | 0 |
| natural-order violation images / rows | 25 / 33 |
| positive / preservation event contexts | 440 / 1,214 |
| generated tokens / model forwards | 29,686 / 29,686 |
| model constructions / natural decodes | 1 / 256 |
| peak GPU reserved | 12,721,324,032 bytes |
| peak host RSS | 9,941,123,072 bytes |
| decode / total wall time | 2,341.49 / 2,362.80 seconds |

The four non-natural stops are included within the eight excluded images; they
are not an additional loss from the 248-image eligible denominator.
Natural-order violations were monitors over the eligible denominator. They did
not exclude an image, invalidate a preservation row, or fire a scientific
stop. All forbidden-work counters were zero, including deficits, gradients,
K4/K2, optimizer steps, checkpoints, dense payloads, and merge exports.

## Why the K1 null is impossible

The registered null requires a complete permutation within each exact
canonical `(category-token tuple, row length)` stratum. Every assigned bundle
must come from a different image, have disjoint owners, have no fixed point,
and retain the exact canonical multiset. Subset shrinkage, resampling, and
partial bundles are forbidden.

For one such stratum, let `N` be its event count and `n_g` the number of events
from image `g`. Because owner identities are image-scoped, the admissible
bipartite graph connects every event to every bundle outside its own image.
Hall's theorem then gives the exact criterion

\[
\text{a complete different-image assignment exists}
\quad\Longleftrightarrow\quad
\max_g n_g \leq N/2.
\]

Necessity follows because the largest image group has only `N - n_g` bundles
outside that image. For sufficiency, any left subset spanning at least two
image groups sees all right vertices; a subset inside one group has at most
`n_g <= N - n_g` members and sees exactly those `N - n_g` outside vertices.

The receipt contains `101` infeasible strata affecting `157` of the `440`
positive events. Of these, `80` are singleton strata and `89` contain events
from only one image. Stratum sizes are
`{1:80, 2:6, 3:6, 4:5, 5:1, 7:2, 8:1}`; unique-image counts are
`{1:89, 2:10, 3:1, 4:1}`. All `101 / 101` violate
`max_g n_g <= N/2`; there are no unexplained infeasible strata. Since the
preflight already fails after collapsing all continuous features to one bin,
the finer registered K4/K2/K1 sequence cannot restore feasibility.

## Interpretation and claim boundary

**Observed:** the frozen 256-image cohort supplies many actual-prefix positive
and preservation events. Natural output is sometimes unsorted, but the
corrected monitor-only semantics preserve those images and Source rows.

**Inferred:** the exact null over-stratifies sparse image-scoped events. The
registered complete permutation fails for a structural image-multiplicity
reason before optimization begins.

**Not supported:** this run says nothing about DoRA fit, semantic gradient
alignment, held-out transfer, annotated-owner generalization, precision,
hallucination, full-scene completeness, or production readiness. It also does
not compare D0 with censored transcript learning.

## Stop and possible successor

The registered route is closed. G0.3 and G1 must not be launched from this
unit. Replacing the exact null changes the estimand and requires a new or
amended research phase owned by the user.

The cheapest proposed successor is a cross-image min-cost perfect transport
null: keep category-token tuple and row-length marginals exact, forbid
same-image assignments, and minimize mismatch in the continuous difficulty
features instead of requiring exact bin equality. A CPU-only feasibility and
balance screen should precede any model load. Two weaker alternatives are a
maximum-cardinality weighted null with an explicitly changed overlap estimand,
or cross-fitted nuisance residualization; both introduce more variance or
assumptions. None is authorized or executed by this result.
