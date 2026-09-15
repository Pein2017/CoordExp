---
title: Fixed-Prompt Clean-versus-Degraded Coordinate Branch Replication Results
type: investigation
role: research-results
authority: executed_evidence
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Fixed-Prompt Clean-versus-Degraded Coordinate Branch Replication Results

## Verdict

The person-25 observation partly replicates, but it is not one universal
failure mechanism.

Across three crop-reviewed same-object cases, the raw probability in a narrow
physical-boundary neighborhood is larger than the probability of the single
coordinate argmax in all three cases. This establishes that physically useful
coordinate support can be distributed across several tokens instead of being
represented by one winning token. However, after applying the source rollout's
temperature and nucleus-sampling cutoff, the tight neighborhood remains
clearly available in only one of the three primary cases. The predeclared
combined gate therefore **fails**: its raw-mass condition passes in all three
cases, but its policy-survival condition passes in only one. This is not a
general decoding prescription.

The six cases reveal three different within-row behaviors:

1. one boundary can fail while the later boundaries remain attached to the
   same object;
2. a small early coordinate difference can redirect later coordinates toward
   a truncated extent or a different nearby instance;
3. a sampled bad extent can come from a low-probability tail even when the two
   conditioned distributions remain nearly unchanged.

The negative controls also reject a universal claim that description or `x1`
already fixes a complete physical-object owner. In the strongest nearby-person
control, the shared `x1` distribution is ambiguous, `y1` begins the route
change, and `x2` and `y2` then cleanly separate the two people.

This result keeps coordinate- and whole-row calibration open as a credible
future research target. It does **not** justify independent coordinate
averaging, a blanket marginal decoder, an explicit object slot, a covered-set
carrier, or training from this unit.

## Executed evidence

The primary model was Qwen3 Vision-Language 2B with the geometry-sorted
Gaussian coordinate-target Weight-Decomposed Low-Rank Adaptation (`DoRA`)
adapter at checkpoint step `4887`. The source pairs came from independent
full-image rollouts with a fresh prompt per call, temperature `0.4`, nucleus
sampling cutoff (`top-p`) `0.95`, and repetition penalty `1.0`.

The frozen case set contains three same-object extent cases and three
same-class owner-switch controls. Every source pair passed checks for source
arm, image, prompt token identifiers, model composition, row index, category,
and exact replay image bytes. Crop-enlarged overlays were inspected before the
cases were admitted.

The primary scoring arm used brain floating point 16-bit (`bfloat16`) model
forward computation to approximate the historical runtime and converted logits
to 32-bit floating-point probabilities. A separate full-model 32-bit
floating-point arm measured precision sensitivity. Scaled dot-product
attention (`SDPA`) matched the source runtime. Each arm scored all four
coordinate positions under both the clean and degraded teacher-forced branch
prefixes, for 48 short forward states per arm.

Artifact roots:

- primary matched-precision scores:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/coordinate-branch-bfloat16-forward-float32-probability-v1/results.json`;
- full-model 32-bit floating-point control:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/coordinate-branch-float32-v1/results.json`;
- true greedy full-row run:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/qwen3-vl-2b-desc-first-geometry-sorted-gaussian-dora-step4887-fixed-prompt-coordinate-branches-20260718T171806Z`;
- crop-enlarged review overlays:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/review/crops/`.

The executing source revision was `0998687e173f2e41bf74e611a9c6ce477c31a60b`.
The case manifest, scorer, and inference configuration Secure Hash Algorithm
256-bit (`SHA-256`) hashes were, respectively, `c235327d...89dc`,
`53714776...73d4`, and
`f3357238...1bee`. The primary and precision-control result hashes were
`229fc106...f205` and `92dfff45...523f`.

## Primary same-object cases

All probabilities below are full-vocabulary probabilities at the causally
relevant coordinate state. `Reference +/- 4` means the sum over the accepted
physical boundary bin and four coordinate bins on either side. The policy
column applies temperature `0.4`, top-p `0.95`, and renormalization to the
replayed logits.

| Image and coordinate state | Coordinate argmax and raw probability | Raw reference +/- 4 mass | Policy reference +/- 4 mass | Interpretation |
|---|---:|---:|---:|---|
| `15335`, person `y1`, shared prefix | bin `0`, `0.09276` | `0.16542` | `0.08744` | Independent top-boundary error. A valid cluster survives the source policy even though the canvas edge is the single-token winner. |
| `16451`, umbrella `y2`, clean prefix | bin `283`, `0.02255` | `0.03996` | `0.00000` | Raw canopy-boundary support is distributed, but the tight official-boundary cluster is removed by top-p. The clean and degraded conditioned distributions remain similar, so the sampled bin `315` is better explained as a low-probability tail plus extent-definition ambiguity than as branch amplification. |
| `7574`, bowl `y2`, clean prefix | bin `188`, `0.04853` | `0.27767` | `0.53311` | The coherent branch strongly supports the full bowl extent. |
| `7574`, bowl `y2`, degraded prefix | bin `172`, `0.04961` | `0.08787` | `0.00000` | An earlier one-bin `x1` branch followed by a different `x2` prefix shifts and sharpens the final lower-boundary distribution toward a truncated bowl. The tight valid cluster is then cut off by top-p. |

The window-width sensitivity matters. At plus or minus eight bins, the policy
mass is `0.19891` for the person, `0.00905` or `0.00258` for the umbrella, and
`0.08128` for the degraded bowl. Wider windows recover still more mass, but
they increasingly mix legitimate boundary uncertainty with different extents
or owners. The result is therefore evidence for fragmented raw support, not
evidence that any chosen neighborhood is automatically a safe decoding unit.

### Image 15335: one bad axis does not destroy the object

The clean and degraded rows are `[821,44,884,136]` and
`[821,0,888,146]`; the accepted reference is `[837,45,884,143]` in
coordinate-bin space. The first difference is `y1`. Under the bad `y1=0`
branch, later `x2` and `y2` still retain substantial mass around the small
person. This is a direct example of an axis-specific boundary failure rather
than an atomic switch to another object.

The unconstrained greedy rollout does not select this small person at row
zero; it emits `[573,27,999,999]` for another large person extent. That greedy
row is a selection control, not a contradiction of the fixed-row result.

### Image 16451: extent convention and tail sampling

The two umbrella rows share `x1=248` and differ mildly at `y1` and `x2` before
ending at `y2=280` versus `315`. The accepted canopy boundary is `260`. Both
conditioned `y2` distributions prefer bin `283`; the bad bin `315` has low
probability and does not become a new mode under the degraded prefix. The
greedy full row `[249,197,472,283]` lands near the clean branch.

This case warns against calling every larger box an instance-binding failure.
Whether the physical umbrella extent includes only the canopy or more of the
pole is an annotation and ontology choice, while the very low-probability
`315` sample remains a sampling-tail event.

### Image 7574: early branch amplification

The clean and degraded bowl rows are `[162,145,238,188]` and
`[161,145,231,164]`; the accepted bowl reference is `[161,156,240,186]`.
Both first coordinates are physically plausible. Under the clean prefix, the
final `y2` argmax is `188` and the tight reference band owns `0.27767` raw
mass. Under the degraded prefix, the argmax shifts to `172`, the tight mass
falls to `0.08787`, and top-p removes that tight band.

The unconstrained greedy row is `[161,145,234,172]`: it combines the shared
left and top boundaries, an intermediate right boundary, and the degraded
lower-boundary mode. This is direct behavioral evidence that a row can be
assembled sequentially from partially different geometry basins rather than
chosen as one indivisible full box.

## Same-class owner controls

The controls show why independent coordinate marginalization would be unsafe.

- **Image 12670, nearby people.** At shared `x1`, the argmax is `418`, between
  clean `414` and degraded `422`, and the reference plus-or-minus-eight mass is
  `0.38628`. This does not lock an owner. Under the clean branch, `x2` reference
  mass is `0.92296`; under the degraded branch it falls to `0.00318` while the
  other person's `x2=474` becomes the argmax. The owner separation emerges
  through `y1` and becomes decisive at `x2` and `y2`.
- **Image 6471, adjacent spectators.** `x1` is precision-sensitive and heavily
  influenced by the canvas edge. The two prefixes then select different
  spectator corridors at `x2`. A coordinate-wise average would fall between
  people rather than represent either person.
- **Image 15517, neighboring buses.** The first-coordinate distribution is
  broad and the greedy row `[479,532,624,585]` combines the degraded left edge
  with right and lower edges close to the clean bus. This is owner ambiguity or
  a mixed bus extent, not clean evidence that one early token owns the row.

The full-model greedy control selects a near-clean person on image `12670`, a
near-clean umbrella on image `16451`, a mixed bowl branch on image `7574`, a
mixed bus branch on image `15517`, and different or neighboring people on
images `6471` and `15335`. Greedy decoding therefore does not merely reproduce
one of the two sampled source rows.

## Precision sensitivity

The matched-runtime `bfloat16` arm is the primary replay. The full-model
32-bit floating-point control agrees on the coordinate argmax in `34/48`
teacher-forced states. The median absolute argmax-bin shift is `0`, and the
mean absolute coordinate-entropy difference is `0.0323` natural-log units.

The largest argmax changes occur mainly in already ambiguous controls, such as
the top boundary of a nearby person, the canvas-edge spectator, and the bus
left boundary. The three primary mechanisms remain qualitatively unchanged.
This makes the result robust enough for the bounded conclusions above, while
also showing that the transformed replay distribution is not a byte-exact
reconstruction of every historical batch-four sampling call.

## Comparison with the person-25 anchor

The person-25 closeout used a different random-order pure cross-entropy
checkpoint and found an unusually strong final-coordinate split: after
temperature `0.4`, the physical person-18 boundary band `514..576` owned
`49.54%` probability while isolated `coord_999` owned `4.288%`.

This six-case replication generalizes the weaker claim that useful physical
boundary probability can be distributed across multiple coordinate tokens.
It does not generalize the person-25 effect size, the exact final-coordinate
location, or guaranteed survival under top-p. The person-25 result is therefore
not a pure anomaly, but it is a particularly clean and strong member of a more
heterogeneous family.

## Hypothesis decisions

| Explanation | Decision | Evidence |
|---|---|---|
| Physically valid boundary support can be distributed across coordinate tokens | Bounded support | Raw reference plus-or-minus-four mass exceeds the isolated raw coordinate argmax in all three primary focus states. |
| The predeclared raw-mass plus policy-survival gate passes | Rejected | Only image `15335` retains clear tight-band support after the source policy; images `16451` and `7574` have zero tight-band policy mass at the failure-relevant state. |
| The source sampling policy always retains the tight valid cluster | Rejected | The tight cluster is removed for the umbrella and degraded bowl; conclusions change with window width. |
| Description or `x1` universally fixes one complete physical-object owner | Rejected as a universal claim | Nearby-person ownership separates after `x1`; the bowl and bus greedy rows combine branch components. |
| Earlier coordinate choices can causally reshape later geometry | Supported in at least one primary case | The bowl's clean and degraded prefixes produce sharply different `y2` distributions and a mixed greedy row. |
| Every bad extent is caused by earlier autoregressive drift | Rejected | The umbrella conditioned distributions remain similar; its bad extent is a low-probability tail and ontology-sensitive case. |
| Every coordinate error is an owner switch | Rejected | Image `15335` preserves the same object after an erroneous top boundary. |
| The person-25 mechanism is unique | Rejected in its weak form; retained in its strong form | Distributed raw mass recurs, but the person-25 magnitude and clean policy survival do not recur universally. |

## Supported

- Geometry errors must be analyzed per coordinate and per branch prefix, not
  only through final-box Intersection over Union.
- Some failures are probability-calibration problems around a physical
  boundary; others are trajectory-dependent owner or extent changes.
- A useful treatment must preserve whole-object consistency while changing
  coordinate probability, because independent coordinate repair can mix
  neighboring instances.

## Ruled out

- One monolithic mechanism explains all clean-versus-degraded sampled boxes.
- `x1` is a universal completed instance-binding point.
- A raw neighborhood-mass result alone licenses coordinate averaging or
  sampling from a post-hoc physical-boundary window.

## Unresolved

- Whether a learnable row-level objective can move valid distributed support
  into a coherent greedy box without damaging object selection.
- Whether the same behavior is prevalent beyond the six deliberately selected
  cases.
- Whether geometry-sorted and random-order adapters develop different rates of
  branch amplification; only the geometry-sorted adapter was tested here.
- How much accepted-boundary uncertainty should be modeled as annotation
  convention rather than prediction error.

## Not claimed

- No conclusion about overall recall, object coverage, stop behavior, mean
  average precision, or the need for a persistent ledger.
- No population prevalence estimate.
- No architecture or training authorization.
- No claim that the four coordinate tokens are statistically independent.

## Next discriminator

Do not expand this probe by default. If training is separately authorized, the
smallest treatment-linked screen is a short same-object coordinate-prefix
consistency and complete-row preference objective:

1. perturb an earlier coordinate only within a visually accepted same-object
   band and require later coordinates to remain on that object;
2. rank one coherent accepted full row above mixed-axis, truncated, expanded,
   and neighboring-instance rows under the same real prefix;
3. retain ordinary row cross-entropy and measure preservation of greedy object
   selection, valid rows, and native one-shot behavior.

The screen should not average coordinate distributions independently and
should not introduce slots or a covered-set carrier. A positive result would
support loss-side calibration of the existing autoregressive path. Failure
despite visible coherent-row probability would raise the priority of a
phase-specific visual or state intervention.
