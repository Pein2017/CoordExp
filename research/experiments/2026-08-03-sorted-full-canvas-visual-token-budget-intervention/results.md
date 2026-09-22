---
title: Sorted Full-Canvas Visual-Token-Budget Intervention - Results
description: Verified invalid-arm result for the paired two-times full-canvas visual-token-budget intervention on Sorted step-4887.
type: investigation
role: research-results
authority: non_normative_research
unit_id: 2026-08-03-sorted-full-canvas-visual-token-budget-intervention
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-03
---

# Results

## Verdict

The execution and evidence chain are valid, but the treatment arm is invalid
for mechanism interpretation.

The approximately two-times full-canvas visual-token budget produced nominal
treatment-calibrated support for `14/63` persistent owners, including `11/51`
`person` owners across seven images. These nominal gains are scientifically
void because the treatment failed the preregistered resolved-owner retention
gate:

- confirmation native-true-positive retention was `64/71 = 0.9014`, passing
  the `0.90` gate by one owner; and
- resolved-owner retention was only `86/114 = 0.7544`, below the required
  `0.90`.

The failure is substantive. Independent recomputation reproduced the
thresholds, every gain and loss, all artifact seals, and the final gate. Even a
non-authoritative audit union that credited every observed adjacent-context
relocation and every loss passing the predecessor threshold reached at most
`98/114 = 0.860`, still below the frozen gate.

The bounded conclusion is therefore:

> Under the frozen Sorted step-4887 category-conditioned scoring interface,
> deterministic full-canvas densification from roughly one to two thousand
> merged visual tokens did not yield an interpretable recovery result. The
> intervention materially changed support dispositions for known-supported
> owners, so no nominal persistent-owner gain may be attributed to increased
> token density.

This closes the full-canvas visual-token-density route under this interface.
No third scale, crop, bank widening, threshold retuning, scale-targeted
training, or architecture escalation follows from this unit.

## Executed evidence boundary

| Item | Executed result |
| --- | --- |
| Checkpoint | Geometry-sorted pure-cross-entropy/type-gate step `4887`, HF full-model fp32, repetition penalty `1.0` |
| Panel | Frozen human-refined twelve-image panel |
| Baseline evidence | Predecessor census run `20260803T065743Z` at about `1024` merged visual tokens |
| Treatment | Same raw COCO JPEGs and deterministic resampler, `max_pixels = 32 * 32 * 2048`, about `1944` merged visual tokens on the representative grid |
| Run root | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-full-canvas-visual-token-budget-intervention/20260803T161241Z/` |
| Analysis | `analysis/intervention-report.json` |
| Treatment calibration | Discovery native true positives, `n=70`, arm-local `q10` plus epsilon `0.002` |
| Capture | `12/12` complete shards, `1,211` query groups, `167,579` score rows |
| Capture seals | Aggregate SHA-256 `ca822d8f659af7573dbc13c45982e9a1863a18ac1de37a3e26f655dedda1cbbc` |
| Paired atlas | `visual/paired/visual-manifest.json`, `43` figures |

The capture used the real Hugging Face model, score-only execution, candidate
batch `16`, and no free decoding. All shard identities were uniform. Every
batched-versus-scalar maximum difference was below `1e-4`. The analyzer sealed
the exact receipt and score bytes it consumed. CPU-only full reanalysis and an
independent contract audit reproduced the published report.

Raw log probabilities are not compared across arms. Each arm uses its own
native-true-positive calibration; the paired atlas normalizes confidence
within each arm only.

## Primary and retention outcomes

| Outcome | Result | Frozen requirement | Disposition |
| --- | ---: | ---: | --- |
| Persistent-owner nominal recovery | `14/63` (`22.2%`, Wilson 95% interval `13.7%–33.9%`) | at least `13/63` | nominally passes |
| Person-only nominal recovery | `11/51` (`21.6%`, Wilson 95% interval `12.5%–34.6%`) | at least `10/51` | nominally passes |
| Image spread | `7` images | at least `3` | nominally passes |
| Confirmation native-TP retention | `64/71` (`90.1%`) | at least `90%` | passes by one owner |
| Resolved-owner retention | `86/114` (`75.4%`) | at least `90%` | **fails** |
| Restricted image `4134` | `3/9` nominal recovery | descriptive only | cannot decide the mechanism |

The final preregistered conjunction is false because both retention gates did
not remain intact. Leave-one-image-out sensitivity also shows image dependence:
holding out image `16228`, `1584`, or `7511` breaks at least one nominal
recovery count requirement.

## Arm-invalid diagnostics

The following observations describe why the arm failed. They do not rescue the
mechanism claim and are not alternative gates.

- Persistent up-flips were `14/63` while known-supported controls produced
  `28/114` resolved down-flips and `7/71` native-true-positive down-flips.
- Gains and losses co-located. Images `16228` and `7511` supplied `8/14`
  nominal gains and `15/35` combined control losses.
- The flips were not cleanly size-conditioned. Recovered persistent owners had
  median minimum dimension `17.5` pixels versus `16.0` for unrecovered owners;
  lost resolved owners had `30.5` pixels versus `30.0` for retained owners.
- Nine of the 28 resolved losses retained treatment-calibrated support at a
  scored but retention-ineligible adjacent context. This is a lower-bound
  indication that some support migrated along the trajectory, but it does not
  repair the frozen retention failure.
- Paired local visualizations commonly retain peaks in the same physical
  neighborhood while changing their rank, extent, or concentration. They are
  consistent with global presentation-dependent reweighting rather than a
  uniform appearance of previously missing local evidence.

The strongest simple explanation is **global presentation perturbation**:
changing the visual grid re-tokenizes the complete image and alters the
category-to-coordinate landscape through the multimodal positional and
decoder interface. A small-owner benefit masked by that perturbation remains
logically possible, but this arm cannot identify it and the descriptive size
analysis provides no positive support for it.

## Observed

- The treatment capture, identities, score rows, calibration, denominators,
  and final gate are reproducible.
- The denser full-canvas presentation changes calibrated owner-support
  dispositions in both directions at fixed model weights.
- Frozen-context support is not presentation-robust enough to serve as a
  treatment admission signal without an independently validated preservation
  criterion.

## Supported

- The unit's preregistered preservation risk is real: a whole-image visual-grid
  change can globally perturb the category-conditioned coordinate field.
- The predecessor census remains valid. This successor does not alter its
  finding that `114/202` eligible false negatives have usable localization
  support somewhere under the original presentation.

## Ruled out or demoted

- Approximately two-times full-canvas densification is ruled out as a
  calibration-preserving mechanism discriminator under the frozen interface.
- Full-canvas patch-sampling scarcity is demoted as the leading bounded
  explanation for the persistent cohort; its cheapest declared intervention
  did not isolate a size-conditioned effect.
- The nominal `14/63`, `11/51`, and restricted `3/9` results cannot be cited as
  evidence that density rescued those owners.

## Unresolved

- Whether some small persistent owners benefit from denser sampling when a
  presentation-robust support functional is available.
- Whether the remaining persistent owners lack usable fixed-resolution visual
  information or are inaccessible through the category-to-coordinate query.
- Why the majority cohort of `114` owners with tested support still fails
  natural greedy rollout.
- Which simple route, ranking, or prefix intervention can convert that support
  into gained and retained physical owners without exchanging prior owners.

## Not claimed

- No natural greedy rollout, stopping, duplication, or final-set result.
- No claim that the vision tower lacks information.
- No claim about crop-and-upscale, new sensor detail, a different
  densification policy, or an impossible owner.
- No cross-arm raw-likelihood comparison.
- No training objective, explicit state carrier, commit token, contrastive
  binding loss, detector, or architecture is promoted.

## Independent review

The independent contract audit found no decision-bearing discrepancy. The
independent scientific audit reproduced the report from sealed bytes and
classified the arm as scientifically invalid under the frozen rule. It also
found that the resolved-retention failure remains below `0.90` under every
generous descriptive re-accounting it tested.

The reviewer identity, evidence boundary, reproduced checks, and route-narrowing
advice are preserved in [review.md](review.md).

The only process residue was stale lifecycle/checklist metadata, corrected in
this closeout. No fresh GPU re-attestation was required because the executed
smoke and capture receipts already owned the runtime gates.

## Next discriminator

First use the predecessor presentation and existing artifacts to describe the
`114` supported-but-native-missed owners:

- whether their supporting context lies on the native greedy trajectory;
- whether the owner ever wins its category group at such a context;
- how often support exists only at a context the natural route has already
  crossed or never reaches; and
- how those cases distribute across route, local competition, and boundary
  gate conditions.

This is a CPU-only prevalence analysis, not a new owner classifier or a
training admission rule. It returns the program to the majority false-negative
regime and the original natural-rollout question.

After that description freezes the remaining ambiguity, the smallest new
GPU-bearing discriminator is the already named image-`4134` fixed-prefix
route/repetition probe. Repetition penalty `1.0` remains the causal primary;
`1.10` may be a separately declared broad trajectory-actuation control, never
a duplicate-only mechanism proof. Complex commit/binding/contrastive training
remains held until simpler reachability and route controls fail and a stable,
owner-selective causal state is demonstrated across images.
