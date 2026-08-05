---
title: Sorted Owner Accessibility Phenotype Census - Results
description: Verified twelve-image census of category-conditioned localization support, proposal surfaces, and one discovery-authored same-description crowding phenotype on Sorted step-4887.
type: investigation
role: research-results
authority: non_normative_research
unit_id: 2026-08-03-sorted-owner-accessibility-phenotype-census
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-03
---

# Results

## Verdict

The census closes with verified observational evidence over all `346` physical
owners in the frozen twelve-image panel.

The most decision-relevant result is not that all false negatives are visually
absent. Among the `202` native-false-negative owners eligible for a conclusion:

- `114/202` (`56.4%`) have calibrated category-conditioned localization
  support at their geometry in at least one tested non-loop native context;
- `72/202` (`35.6%`) have no tested localization support under the frozen
  17-role owner-centred bank, canonical category query, and tested native
  contexts; and
- `16/202` (`7.9%`) flip across the ambiguity bounds and remain unresolved.

This divides the original false-negative question into at least two empirical
regimes. The `114` supported-but-native-missed owners are plausible
selection/routing/traversal failures and therefore credible training-side
targets. The `72` persistent owners are not licensed as "the visual tower did
not see them"; the permitted statement is only that this fixed interface found
no usable local category-field peak at their geometry.

The single discovery-authored crowding phenotype reproduced only
directionally on the held-out half. It passed the preregistered coverage gate
but not a strong separation bar. It is an enrichment descriptor, not an
owner-level explanation or a causal mechanism.

No training objective, architecture, detector, annotation mutation, or
inference policy is promoted by this unit.

## Executed evidence boundary

| Item | Executed result |
| --- | --- |
| Checkpoint | geometry-sorted pure-cross-entropy/type-gate step `4887`, repetition penalty `1.0` |
| Panel | `12` human-refined images; `346` owners |
| Split | discovery `[10707, 14038, 2685, 5001, 6040, 7511]`; confirmation `[13348, 13923, 14439, 1584, 16228, 4134]` |
| Contexts | root plus every complete native row boundary plus terminal; forced-continue rows excluded |
| Candidate bank | `17` score-independent owner-centred roles, alias-collapsed; exact GT anchor mandatory |
| Query | `[OBJECT_REF_START, category tokens, OBJECT_REF_END, BOX_START]` |
| Run root | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z/` |
| Capture manifest | `baa4081fdc072f22ba6a4cf705574a957f88f492f370cca4ddf2214c5b8a8df5` |
| Calibration | `9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5` |
| Frozen discovery rule | `45bbe07065670a4291ed7d874fc2a8ca15caee79adf69e3f5fc78ab2e43101f8` |
| Capture validity | `12/12` complete shards; `0` quarantined, missing, incomplete, or subset shards |
| Visual product | `visual/combined/visual-manifest.json`; `61` figures over five product types |

All twelve shard receipts bind the same model, tokenizer, planner, scorer, and
runtime identity. Candidate batching was `16`; every shard passed its
batched-versus-scalar and argmax checks, and all decision-bearing rows were
admission-covered. The largest shards (`14038` and `4134`) completed without
quarantine. The GPU capture is finished and the GPUs were released before
analysis.

## Calibration and dispositions

Discovery native true positives at their deterministic due boundary produced
the pooled `q10` thresholds:

| Statistic | Threshold |
| --- | ---: |
| `peak_lift` | `2.3884847780085985` |
| `local_concentration` | `1.63233060836792` |
| Calibration observations | `70` |
| Single-context epsilon | `0.002` |

Support requires both statistics to clear threshold plus epsilon. Rank and
margin remain routing/competition diagnostics and never decide support.

| Disposition | Discovery | Confirmation | Combined |
| --- | ---: | ---: | ---: |
| Native-TP calibration control | `70` | `71` | `141` |
| Resolved tested localization support | `54` | `60` | `114` |
| Persistent no tested localization support | `38` | `34` | `72` |
| Ambiguity-bound disposition flip | `12` | `4` | `16` |
| Outside native matching universe | `3` | `0` | `3` |

The confirmation half alone therefore places `60/98` (`61.2%`) eligible
false negatives in the resolved-support cohort, `34/98` (`34.7%`) in the
persistent cohort, and `4/98` (`4.1%`) in the ambiguity-neutral cohort.

## Frozen crowding phenotype

After discovery, exactly one rule was sealed:

```text
view       = primary_first_non_loop_minimal_abs_frontier
bound      = U
stratum    = native FN, greedy-eligible, frontier-tested
condition  = same_description_owners_ahead_of_frontier >= 8
```

The rule is score-independent. A tempting alternative based on
`margin_to_best_owner_in_group` was rejected because discovery Pearson
correlation with `peak_lift` was about `0.935`; it would largely re-derive the
support outcome it purported to explain.

### Discovery

Among owners with a closed resolved-versus-persistent disposition:

| | Persistent | Resolved | Persistent share |
| --- | ---: | ---: | ---: |
| Crowded | `29` | `19` | `60.4%` |
| Sparse | `9` | `35` | `20.5%` |

Relative risk was `2.95`; one-sided Fisher exact `p = 9.37e-5`.

### Held-out confirmation

| | Persistent | Resolved | Persistent share |
| --- | ---: | ---: | ---: |
| Crowded | `18` | `23` | `43.9%` |
| Sparse | `16` | `37` | `30.2%` |

Relative risk was `1.45`; one-sided Fisher exact `p = 0.124`. The `18`
crowded persistent owners span four confirmation images, satisfying the
preregistered minimum coverage of at least `15` false-negative owners in at
least four images. The effect nevertheless attenuated sharply and does not
support a strong predictive or causal claim.

The bounded conclusion is therefore **directional replication with weak
held-out evidence**, not full confirmation. The strongest alternative is
category/image composition: the discovery persistent cohort was dominated by
`person` owners and image `7511`.

## Post-confirmation exploratory scale signal

After the rule was applied unchanged, a new score-independent geometry analysis
compared owner size between the closed resolved and persistent cohorts. This
was not preregistered and is a next-unit hypothesis, not a confirmed phenotype.

| Quantity | Persistent (`n=72`) | Resolved (`n=114`) |
| --- | ---: | ---: |
| Median normalized bbox area | `0.000424` | `0.001781` |
| Median minimum box dimension | `17.5 px` | `30 px` |
| Owners with minimum dimension `<16 px` | `29/72` | `13/114` |

Normalized area separates resolved from persistent with area under the
pairwise ranking curve `0.753` over all twelve images (`p=3.31e-9`) and
`0.789` on the held-out confirmation half (`p=1.75e-6`). The same direction
remains within `person`: `0.699` over all images and `0.779` on held-out
persons. By contrast, the frozen crowding cut does not separate persistent
from resolved within the held-out `person` subset (one-sided Fisher
`p=0.709`).

This makes apparent object scale the strongest current alternative to the
crowding explanation. It still does not prove a vision-tower information
limit: small geometry is correlated with blur, occlusion, same-category
density, annotation extent, and visual-token resolution.

Independent post-confirmation review reproduced the same signal within
non-`person` owners (`AUC=0.778`) and found the per-image direction in `10/12`
images. It also found real residues: `5001` and `6040` reverse the direction in
very small samples, while loop-degenerate `4134` is scale-flat among persons
(`AUC=0.500`) and contributes `9/34` confirmation persistent owners. Queue
length and log owner area correlate at about `-0.49`, so crowding was plausibly
a noisy proxy for scale and category composition. Scale is therefore the
leading explanation for much of the cohort, not all of it.

## Observed

- More than half of native false negatives retain category-conditioned local
  geometry support somewhere on their native trajectory.
- More than a third do not exhibit such support under the fixed tested
  interface.
- Continue/stop gate weakness is not the leading discriminator at each owner's
  best localization context; the gate is usually open in both cohorts.
- The freely generated sidecar almost never strictly recovers discovery
  persistent owners (`1/38`), while it recovers `16/54` discovery resolved
  owners and `69/70` discovery native-TP controls. Sidecars remain diagnostic
  only and never change a disposition.
- The atlas visually confirms that the persistent-heavy images contain many
  very small or distant instances; it does not convert that visual impression
  into a causal claim.

## Supported

- Dense-enumeration false negatives are heterogeneous: a substantial
  supported-but-unselected cohort exists alongside a persistent-no-tested-peak
  cohort.
- The supported cohort is a credible target for selector/routing/traversal
  training because a usable coordinate landscape already exists under some
  native contexts.
- Same-description queue length is a discovery-enriched phenotype with only a
  weakened held-out directional replication.
- Apparent owner scale is the strongest post-confirmation candidate
  discriminator for the persistent cohort and should own the next probe.

## Ruled out or demoted

- A single universal "the model never saw the object" account is ruled out by
  `114` supported native false negatives.
- Boundary STOP alone is demoted as the primary explanation for these owners.
- Category-routing rank alone is demoted for the persistent discovery cohort;
  their category frequently remained highly routed while local geometry did
  not.
- The frozen crowding threshold is not promoted as an owner-level classifier,
  causal mechanism, or training admission rule.

## Unresolved

- Whether the `72` persistent owners lack usable information in the frozen
  visual representation, or whether the category-to-coordinate interface and
  finite bank fail to expose that information.
- Whether scale itself is causal or proxies blur, occlusion, dense
  same-category competition, or annotation-extent ambiguity.
- Which intervention best converts the `114` supported-but-missed owners into
  retained natural-rollout owners without exchanging previously recovered
  owners.

## Not claimed

- No claim that the vision tower lacks information.
- No claim that crop-and-upscale recovery would diagnose the frozen full-image
  representation.
- No raw log-probability comparison across images.
- No per-owner proposal probability.
- No causal or architecture conclusion.
- No claim that a persistent owner is impossible to recover with a broader
  query, denser spatial bank, different representation readout, or image
  resolution.

## Next discriminator

Both halves of this panel are now spent for observational rule selection. The
next live unit should therefore be a prospectively frozen **full-canvas
apparent-resolution intervention** on balanced persistent and resolved owners,
with `person`-only and within-image matching so category and image identity
cannot explain the result. Change one factor only: increase the full image's
visual resolution under a fixed deterministic policy while keeping checkpoint,
prompt, category query, owner ledger, coordinate scoring interface, and local
bank semantics fixed. Do not use owner crops: crop-and-upscale recovery is
expected and would change spatial context as well as resolution.

The next unit should recalibrate native-TP controls in the intervened condition
and preregister one recovery threshold before scoring. Its stop logic is:

- if small persistent owners gain calibrated local peaks while resolved and TP
  controls retain their support, promote a bounded input-scale-limited support
  mechanism for that cohort;
- if persistent recovery is small or control support is not retained, close the
  full-canvas scale route rather than widening the scale or trying many crops;
  the scale-flat `4134` route/repetition mode then becomes the next
  discriminator; and
- if effects do not survive person-only, within-image matching, keep scale as
  a confound and do not launch scale-targeted training.

The exact minimum recovery and retention thresholds remain proposals for the
next unit and are not retroactively frozen here.

Training on crowding or explicit owner selection should not begin merely from
this census. A later treatment must report gained, retained, and lost unique
physical owners under natural rollout and preserve strict geometry.
