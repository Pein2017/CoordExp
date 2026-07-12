---
title: PVCI Own-prefix Causal Behavior Results
description: Closure note for the 16-cell own-prefix PVCI causal and safety panel.
type: idea
role: research-result
authority: non_normative_research
status: complete
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-causal-proposal-bridge
updated: 2026-07-12
---

# Own-prefix causal behavior results — 2026-07-12

## Closure verdict

The exact artifact/evaluator contract passed, but the causal-use and safety
gates did not. The mechanical evaluator wrote `status=PASS` for artifact
validation and `gate_decision=hold`, with reason
`safety_noninferiority_failed`; its booleans are
`primary_pass=false` and `negative_controls_pass=false`. Applying the unit's
predeclared ordered interpretation to the observed controls gives the earlier terminal label
`shortcut_or_row_prior`: the C-on behavioral signature is reproduced by
another-image and within-image token-permutation feedback, while
position-only feedback is a weaker partial reproduction and norm-matched
random feedback is a null. This is not `causal_bridge_supported`,
`decode_specific_bridge`, or a final-architecture result.

The completed safety implementation also reports `safety_pass=false` against
both A and C-off, including the frozen `GT>=9` duplicate stratum, malformed
delta, and absolute `95%` natural-closure floor.

The training-view exposure mass is separately attested by the executed pack
manifest: `9831` canonical, `1843` jitter, and `614` duplicate-history views.
Under the frozen renderer, terminal EOS is supervised only in canonical full
responses, so the corresponding supervised terminal-EOS atom counts are
`9831/0/0`. These are checkpoint-level training exposures shared by all
held-out images, not per-image labels. The natural own-prefix panel therefore
cannot and does not fabricate STOP strata by teacher-forced exposure; its STOP
metrics are reported by observable held-out count strata instead.

The proposal representation remains a narrow follow-up research handle based
on the separately completed held-out representation gate. The tested
split-layer bridge is not promoted. The result does not show that all
intermediate visual representations fail and does not authorize slots, a
ledger, or any final architecture.

## Exact scope and receipts

This is the metric-bearing natural own-prefix panel for the frozen unit. It
uses the same 320 held-out COCO images and ordered request IDs in every cell:

- launch root:
  `/data/CoordExp/outputs/probes/coordexp_swift/pvci_own_prefix_sharded_panel_ABC_320_v3_20260712/`;
- launch manifest:
  `pvci_own_prefix_launch_manifest.json`, `cell_count=16`,
  `expected_count=320`, `shard_count=8`,
  `manifest_sha256=16cdea3512c1715ed45537a65eb02d7f64f9004d560d22ee4572c76f1bbc9471`;
- cohort source:
  `pvci_proposal_bridge_heldout_preflight/own_prefix_cohort.jsonl`,
  SHA256 `f4f7e792b289105335443de8bc1c0c153ada7c2c98bed25e54eccec2ce971ab5`;
- source rows SHA256
  `f21dc678bda7eb461ffd1b8d4f9d8889b9aafd9da955ff347ba0058ab622a3b1`;
- ordered row/request digest (all 16 cells):
  `1aca065be617de0ac3d5a48f8bfad1d165b95b8da35acf9efa660bfedf34fc68`;
- final evaluator receipt:
  `own_prefix_behavior_gate/own_prefix_eval_receipt.json`,
  `status=PASS`, `condition_count=16`, `request_id_count=320`;
- aggregate and gate artifacts:
  `own_prefix_behavior_gate/own_prefix_aggregate.json` and
  `own_prefix_behavior_gate/own_prefix_gate_decision.json`;
- official secondary artifacts:
  `official_eval/<cell>/{metrics.json,evaluation_receipt.json}`.

The final evaluator was replayed after hardening the safety gate to enforce
both A and C-off baselines, the `9+`-GT duplicate stratum, malformed output,
and the natural-closure floor.  Its PASS receipt now hashes all three emitted
metric/gate artifacts and self-hashes.  The stricter replay preserved
`decision=hold`, `primary_pass=false`, `negative_controls_pass=false`, and
`safety_pass=false`; every safety check is reported separately for A and
C-off.

Executed source-bound trust checks also passed:

- backward/gradient attestation:
  `/data/CoordExp/outputs/probes/coordexp_swift/proposal_bridge_training_backward_attestation/vf4_source_bound_20260712/`;
- runtime feedback attestation:
  `/data/CoordExp/outputs/probes/coordexp_swift/proposal_bridge_runtime_attestation/source_bound_20260712/`.

These receipts rule out a silent wiring or zero-gradient explanation.  They
do not make the learned behavior target-specific.

The panel is A, B, C-off, C-on, C-another-image, C-token-permutation,
C-position-only, and C-norm-matched-random, each at RP1.10 and RP1.00. All
condition receipts are `pvci-own-prefix-condition-v2`, bind the same cohort
and ordered row digest, and passed their shard/merge/provenance checks. C-on
and every C control use the same C checkpoint/config identity within each RP;
C-off is the exact zero-delta orchestration control.

The evaluator uses frozen `src.vis.normalization.load_visual_rows` and
`src.vis.matching.match_row`: exact normalized description, IoU `0.50`,
greedy one-to-one assignment, and duplicate candidates at equal normalized
description plus pair IoU `0.30`. Every bootstrap below is paired at image
level, `10,000` resamples, seed `20260711`, percentile 95% CI. No row-level
bootstrap was used.

## Sixteen-cell own-prefix behavior table

Values are image means over `n=320`. `P` is matched-row precision, `R` is
labeled-GT recall, `UE` is annotated under-enumeration, `Dup` is raw duplicate
rate, `Inv` is the invalid/malformed-row image indicator, `Stop` is premature
annotated stop, `Close` is natural terminal closure, and `Pred` is raw
prediction count per image. `Unknown` extras are retained by the evaluator;
they are not converted to false positives or successes.

| Cell | P | R | UE | Dup | Inv | Stop | Close | Pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A RP1.00 | 0.7604 | 0.7189 | 0.2811 | 0.1208 | 0.0438 | 0.6406 | 0.9813 | 9.80 |
| A RP1.10 | 0.7543 | 0.6887 | 0.3113 | 0.0575 | 0.0156 | 0.6688 | 0.9969 | 8.49 |
| B RP1.00 | 0.7598 | 0.7228 | 0.2772 | 0.1270 | 0.0375 | 0.6344 | 0.9813 | 9.94 |
| B RP1.10 | 0.7522 | 0.6896 | 0.3104 | 0.0597 | 0.0125 | 0.6656 | 0.9969 | 8.55 |
| C-off RP1.00 | 0.7530 | 0.7113 | 0.2887 | 0.1329 | 0.0500 | 0.6375 | 0.9656 | 10.42 |
| C-off RP1.10 | 0.7493 | 0.6866 | 0.3134 | 0.0643 | 0.0250 | 0.6656 | 0.9906 | 8.82 |
| C-on RP1.00 | 0.5426 | 0.7321 | 0.2679 | 0.3365 | 0.3438 | 0.3781 | 0.6688 | 21.44 |
| C-on RP1.10 | 0.5246 | 0.7049 | 0.2951 | 0.2449 | 0.3313 | 0.3563 | 0.6594 | 20.86 |
| C-another-image RP1.00 | 0.5362 | 0.7325 | 0.2675 | 0.3414 | 0.3531 | 0.3750 | 0.6594 | 21.83 |
| C-another-image RP1.10 | 0.5298 | 0.7078 | 0.2922 | 0.2476 | 0.3031 | 0.3906 | 0.6969 | 20.36 |
| C-token-permutation RP1.00 | 0.5707 | 0.7302 | 0.2698 | 0.3053 | 0.3000 | 0.3969 | 0.7031 | 19.92 |
| C-token-permutation RP1.10 | 0.5368 | 0.7068 | 0.2932 | 0.2378 | 0.3219 | 0.3719 | 0.6781 | 20.30 |
| C-position-only RP1.00 | 0.6922 | 0.7201 | 0.2799 | 0.1770 | 0.1094 | 0.5594 | 0.9000 | 12.56 |
| C-position-only RP1.10 | 0.6732 | 0.6974 | 0.3026 | 0.1111 | 0.1063 | 0.5813 | 0.9125 | 12.27 |
| C-norm-random RP1.00 | 0.7533 | 0.7137 | 0.2863 | 0.1117 | 0.0313 | 0.6500 | 0.9938 | 9.56 |
| C-norm-random RP1.10 | 0.7451 | 0.6783 | 0.3217 | 0.0582 | 0.0125 | 0.6781 | 0.9969 | 8.60 |

The striking pattern is not a clean target-specific gain. C-on raises labeled
recall by only about two percentage points over C-off while multiplying raw
prediction volume and invalid/duplicate behavior. Another-image feedback is
nearly indistinguishable from C-on on the harmful precision/validity axis;
token permutation is also close. Position-only is a weaker but still harmful
partial reproduction. Norm-matched random stays near C-off and therefore acts
as a null for this particular pulse magnitude/application path.

## Selected paired bootstrap contrasts

### Primary C-on versus C-off attribution

| RP | Metric | C-on minus C-off | 95% CI |
|---|---|---:|---:|
| 1.10 | matched-row precision | -0.224702 | [-0.256177, -0.194913] |
| 1.10 | labeled-GT recall | +0.018276 | [+0.008195, +0.028785] |
| 1.10 | annotated under-enumeration | -0.018276 | [-0.028785, -0.008195] |
| 1.10 | raw duplicate rate | +0.180594 | [+0.148489, +0.214185] |
| 1.10 | invalid | +0.306250 | [+0.256250, +0.359375] |
| 1.10 | premature annotated stop | -0.309375 | [-0.362500, -0.259375] |
| 1.00 | matched-row precision | -0.210411 | [-0.243699, -0.178149] |
| 1.00 | labeled-GT recall | +0.020786 | [+0.011912, +0.030296] |
| 1.00 | annotated under-enumeration | -0.020786 | [-0.030296, -0.011912] |
| 1.00 | raw duplicate rate | +0.203547 | [+0.166213, +0.241485] |
| 1.00 | invalid | +0.293750 | [+0.243750, +0.346875] |
| 1.00 | premature annotated stop | -0.259375 | [-0.309375, -0.212500] |

Against B at RP1.10, C-on precision is `-0.227649` CI
`[-0.258452,-0.197503]`, recall is only `+0.015280`
`[+0.005680,+0.024846]`, duplicates are `+0.185130`
`[+0.152185,+0.218386]`, and invalid is `+0.318750`
`[+0.268750,+0.371875]`. Thus neither the representation-shaped B arm nor
the C-off checkpoint supports a causal bridge improvement.

### Negative-control attenuation and the row-prior signature

At RP1.10, the control-minus-C-off contrasts are:

| Control | Precision | Recall | Chimera rate |
|---|---:|---:|---:|
| C-another-image | -0.219484 [-0.251107,-0.189206] | +0.021144 [+0.010399,+0.032074] | -0.004186 [-0.013114,+0.004689] |
| C-token-permutation | -0.212486 [-0.243377,-0.182002] | +0.020123 [+0.009389,+0.031148] | -0.004941 [-0.013663,+0.003614] |
| C-position-only | -0.076066 [-0.097292,-0.055287] | +0.010713 [+0.001994,+0.019561] | -0.004073 [-0.010619,+0.002468] |
| C-norm-matched-random | -0.004134 [-0.017537,+0.008343] | -0.008317 [-0.015652,-0.001576] | +0.004044 [-0.000852,+0.010157] |

The evaluator's control rule requires every control's upper CI to be at most
`+0.02` for both precision and recall. That rule does not pass here: the
another-image and token-permutation cells reproduce the same large precision
collapse and recall increase as C-on, and position-only reproduces part of it.
Norm-matched random is the only clear null. In the unit's ordered outcome map,
this makes a row/depth/raster/traversal prior more plausible than target-specific
visual causal use. It does not identify which native state or token position is
the exact source of the prior.

## RP1.10 versus RP1.00

RP1.10 is the predeclared primary decode and RP1.00 the secondary crossover;
neither rescues the bridge. C-on has P/R `0.5246/0.7049` at RP1.10 and
`0.5426/0.7321` at RP1.00, while C-off is `0.7493/0.6866` and
`0.7530/0.7113`, respectively. The harmful C-on-minus-C-off precision
contrast is negative at both penalties. The evaluator records
`rp_contradiction=true` because the RP1.00 precision CI lower bound is below
the predeclared `-0.03` contradiction threshold, but the primary behavior and
safety gates already fail. Therefore this is not labeled
`decode_specific_bridge`; it is a shortcut/safety failure visible under both
decode settings.

## Safety breaches

The frozen safety limits are breached by large margins under both RP values:

- RP1.10 C-on versus C-off: duplicate rate `+18.06` percentage points (limit
  `+3`), invalid/malformed image rate `+30.63` points (limit `+1`), and
  natural terminal closure `65.94%` (floor `95%`).
- In the `GT>=9` stratum, RP1.10 C-on duplicate rate rises `+19.18` points
  versus C-off (CI `[+14.43,+24.04]`) and `+20.68` points versus A
  (`[+15.90,+25.51]`), both far above the `+5`-point dense-scene limit.
- Against A overall at RP1.10, C-on duplicate rate rises `+18.73` points,
  invalid/malformed rate rises `+31.56` points, and closure falls from
  `99.69%` to `65.94%`; safety therefore fails against both required matched
  baselines, not only C-off.
- RP1.00 C-on versus C-off: duplicate rate `+20.35` points, invalid/malformed
  `+29.38` points, and closure `66.88%`.
- C-on emits about `20.86` raw rows/image at RP1.10 versus `8.82` for C-off;
  the excess is not a safe enumeration gain.
- Phrase/geometry chimera increases are not the main breach (RP1.10
  C-on-minus-C-off `-0.007113`, CI `[-0.016775,+0.002082]`), but a benign
  chimera rate cannot offset the duplicate, malformed, and closure failures.

The mechanical gate is consequently `hold`, even though labeled recall and
annotated under-enumeration move in the favorable direction. One seed and one
320-image cohort are sufficient for this route decision, not for a replicated
causal claim.

## Official COCO metrics (secondary, non-benchmark)

The official evaluator artifacts explicitly set `benchmark_eligible=false` and
`benchmark_metric=false`. They are context only and cannot replace the
own-prefix behavior gate or support a COCO headline. Selected all-cell
secondary values are below (`AP`=`bbox_AP`, `AP50`=`bbox_AP50`, `AR100`=`bbox_AR100`):

| Cell | AP | AP50 | AR100 |
|---|---:|---:|---:|
| A RP1.00 | 0.402149 | 0.563478 | 0.462825 |
| A RP1.10 | 0.379150 | 0.533244 | 0.452838 |
| B RP1.00 | 0.400198 | 0.563473 | 0.468743 |
| B RP1.10 | 0.378975 | 0.539573 | 0.455346 |
| C-off RP1.00 | 0.383626 | 0.540011 | 0.450739 |
| C-off RP1.10 | 0.370712 | 0.526445 | 0.451483 |
| C-on RP1.00 | 0.338014 | 0.472633 | 0.470385 |
| C-on RP1.10 | 0.347590 | 0.485439 | 0.463541 |
| C-another-image RP1.00 | 0.324330 | 0.456758 | 0.471109 |
| C-another-image RP1.10 | 0.349298 | 0.488621 | 0.465435 |
| C-token-permutation RP1.00 | 0.348084 | 0.484541 | 0.472611 |
| C-token-permutation RP1.10 | 0.350790 | 0.488112 | 0.467901 |
| C-position-only RP1.00 | 0.384336 | 0.541190 | 0.469930 |
| C-position-only RP1.10 | 0.362477 | 0.512952 | 0.456796 |
| C-norm-random RP1.00 | 0.388993 | 0.543543 | 0.454595 |
| C-norm-random RP1.10 | 0.367535 | 0.519105 | 0.445404 |

The official score rows also show the same non-benchmark warning signs: C-on
has substantially more predictions than C-off, and the controls can have
similar or higher AR100 while precision and validity collapse. These are not
grounds for claiming discovery of unlabeled objects.

## Incomplete-label limitation

COCO annotations are treated as positive current-target supervision, not as a
complete visible-object inventory. `LabeledRowRecall`, `UE`, and
`premature_annotated_stop` refer only to annotated boxes remaining under the
frozen matcher. Unmatched predictions remain `unknown/other`; they are not
automatic hallucinations, and annotation exhaustion is not complete coverage.
The official COCO converter's dropped predictions, parser drops, invalid
geometry, controller-invalid rows, and unmatched extras remain separate
denominators. This limitation prevents a directional claim about all visible
objects or global background. It is recorded as a standing caveat, but it is
not the primary terminal label here because the same harmful behavior is
reproduced by semantic controls and fails explicit safety bounds.

## Hypothesis verdicts

| Hypothesis | Verdict | Evidence-bound interpretation |
|---|---|---|
| H1: proposal is learnable | Supported only at representation handle | The separate held-out v2 gate promoted target-specific proposal information; this rollout does not extend that claim. |
| H2: representation shaping is sufficient | Not supported | B is behaviorally near A; C-off does not deliver a meaningful safe gain. |
| H3: proposal must be causally consumed | Not supported | C-on loses precision and fails safety against C-off/A; null/control attenuation is absent. |
| H4: geo/raster/row shortcut explains behavior | Supported; terminal label | Another-image and token-permutation feedback reproduce the signature; position-only is partial; norm-random is null. Exact native source remains unresolved. |
| H5: positive bag is not instance-specific enough | Unresolved, with concern | Representation specificity passed earlier, but this own-prefix bridge does not bind a safe full row. Do not escalate automatically to slots. |
| H6: proposal feedback is harmful/unstable | Supported for this handle | Duplicates, invalid/malformed rows, prediction bursts, and closure breaches are large and replicated at both RP settings. |
| H7: proposal is not the primary free-rollout bottleneck | Supported for this bridge, not globally | Small labeled-recall movement is dominated by malformed/duplicate behavior. First localize whether any target-specific write survives a short transition; only then choose between a new bridge and COMMIT/COVERAGE/STOP. |
| H8: partial labels limit direction | Limitation confirmed, not terminal | Unknown extras and annotated-only STOP prevent complete-coverage claims, but do not explain the matched control reproduction. |

## Bounded architecture posterior

Posterior update is intentionally narrow:

- **Up:** a supervised proposal readout can carry target-specific visual
  information under held-out teacher-forced controls.
- **Down:** this split-layer text-residual pulse as a metric-bearing causal
  bridge; a claim that C-on safely binds phrase and geometry; and any claim
  that improved labeled recall is a learned object-transition mechanism.
- **Up, but not identified:** a row/depth/raster/traversal prior or other
  position-conditioned decoder pathway can dominate the observed bridge
  response. Another-image and token-permutation reproduction localize the
  result only to the tested delivery surface and controls.
- **Unchanged:** whether a different visual intermediate, a coverage/commit
  state, a better lifetime/cache boundary, or a different objective could
  causally help. Negative evidence here is not evidence that all such
  representations are infeasible.

No slots, entity register, persistent ledger, stronger repulsion loss, or final
architecture is promoted by this note.

## Next research decider

Do **not** start with another 320-image long-rollout panel or a larger module.
The highest-information next probe is a short, controlled row transition that
asks whether the learned proposal can write a target-specific state before the
generic continuation pulse compounds over many rows.

Use examples containing at least two attributable candidate objects.  Hold the
image, legal prior-row prefix, application/reset schedule, pulse norm, and
decode policy fixed.  At the same pre-row boundary compare:

1. the correct proposal-derived pulse;
2. another-image and within-image token-permuted pulses;
3. position-only and norm-matched-random pulses;
4. exact C-off.

Score the first distinguishing phrase token, the complete phrase, the first
coordinate token, the complete box, row validity, and whether phrase and
geometry bind the same candidate.  Free-run at most one current row; do not let
later enumeration/STOP dynamics obscure the intervention.  The gate is
target-specificity, not raw continuation: the correct pulse must move the row
toward its intended candidate while source-swap and token-permutation lose that
effect, with no invalid/duplicate/closure regression.

If this short transition fails, retire the tested split-layer delivery route;
do not tune pulse magnitude or rerun the same long panel.  The next independent
unit should then target native COMMIT/COVERAGE/STOP.  If it passes, the next
question is a write-read intervention: after committing that row, does its
state causally suppress the committed object and redistribute the next native
proposal toward annotated-uncovered candidates?  This remains a route decider,
not a slot or final-architecture commitment.

## Research Unit Closeout

### Observed

All 16 exact own-prefix cells completed over 320 paired images. Artifact
receipts and merged scored artifacts passed validation. C-on raised labeled
recall slightly but caused a large precision collapse, duplicate burst,
invalid/malformed output, prediction-count inflation, and closure failure.
Another-image and token-permutation controls reproduced that signature;
position-only partially reproduced it; norm-matched random did not.

### Supported

The tested proposal-bridge delivery surface is not a safe, target-specific
causal behavior mechanism. The ordered terminal interpretation is
`shortcut_or_row_prior`, with a mechanical evaluator `hold` for safety.
Incomplete-label safeguards remain mandatory.

### Not supported yet

No claim is made for all intermediate visual representations, a different
bridge lifetime, a different causal owner, a coverage ledger, slots, or a
final architecture. No benchmark or complete-visible-object claim is made.

### Next decider

Run the short target-specific transition probe above.  Only a passing result
authorizes the subsequent write-read commit intervention; a failure routes to
a separately bounded COMMIT/COVERAGE/STOP unit rather than enlarging this
bridge.

### Promotion decision

`not_promoted`. The held-out representation handle may be reused in a new,
separately bounded research unit. The causal bridge and any resulting decoder
architecture remain unpromoted.
