# Sorted Owner-Basin Landscape and Repair — Results

## Outcome

The unit closed at the predeclared Task-4 control gate with disposition
`hold`. The twelve-image census, C-sentinel read, repair factorial, and
suppression matrix were not executed because the controls invalidated the
planned adjudication rule before those stages were authorized.

Decision-bearing artifacts:

- Control scores:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-scores-uncached-merged-v1`
- Reconstructed summary:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-summary-uncached-reviewed-v4`
- Calibration and lead review:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v7`

## Observed

- The Task-0 panel contains 346 registered GT owners: 343 were eligible for
  paired decisions and 3 remained globally ambiguous and decision-neutral.
  Within the 343 eligible owners, native greedy strictly matched 141; K16
  sampling reached 204; and 64 owners were strict sampling rescues.
- The merged Task-4 artifact contains 18,448 FP32 score rows. Exact KV-cache
  parity failed the frozen all-vocabulary tolerance, so all decision-bearing
  rows came from literal `use_cache=False` full-prefix reforward. Independent
  reconstruction passed.
- Four restricted GT-target owner/context surfaces became decision-bearing
  after the review-lineage and owner-namespace contract corrections. The four
  canonical-description free surfaces remain raw-only because their generated
  candidates were unreviewed.
- Strict visible control `gt:7511:22` at root had target peak `-16.6701`.
  Target prominence was `+18.9421` over equal-size background and `+10.8998`
  over the registered scan foil. Its frozen shape was
  `part_or_whole_lobes`, not `localized_peak`.
- The strict control's near-peak candidates nevertheless form one tight,
  high-overlap spatial mode. The `part_or_whole_lobes` label was produced by
  the frozen label-first submode partition separating `gt_whole` from adjacent
  `scale_aspect_perturbation` candidates, not by separated owner-level peaks.
- B1 control `gt:7511:26` at root had target peak `-18.3319`, scan-foil
  prominence `+8.4107`, and shape `part_or_whole_lobes`.
- For the B2 pair (`gt:7511:22` already covered, target `gt:7511:26`), the
  target peak changed from `-12.5661` before to `-13.2457` after. Its shape
  changed from `localized_peak` to `part_or_whole_lobes`.
- The registered B2 diagnostic contrasts were mixed rather than jointly
  suppressive:
  equal-size background `25.6573 -> 28.4705`, scan foil
  `19.4302 -> 15.6673`, and covered physical owner
  `-1.3972 -> +0.5241`. Thus the target weakened relative to the scan foil but
  strengthened relative to the background and covered-owner diagnostics.
- The sealed B2 registry did not prospectively bind the required
  same-description, non-overlapping physical control. The v7 receipt therefore
  records `required_matched_physical_foil_absent` and triggers Stop Rule 9.
- The mechanical lead receipt records Stop Rules 3 and 9 as `triggered`, no
  sentinel read, no non-C freeze, and disposition `hold`.

## Supported

- The restricted, owner-conditioned landscape exposed substantial localized
  target support in the strict positive control.
- The frozen `localized_peak` operationalization is not valid for adjudicating
  this unit: its label-first submode partition can split one tight spatial peak
  into multiple registered submodes. This is a classifier defect, not evidence
  that the strict control has multiple owner-level or spatial lobes.
- The single B2 control does not support a clean same-description collision
  claim. Its required physical matched control was absent; the available
  post-hoc diagnostic contrasts were also mixed.
- Existing Task-4 GPU work was scientifically recoverable. The initial null
  summary came from two contract bugs: sealed/reviewed lineage was treated as
  unreviewed, and target owner IDs were compared against a diagnostic-ID
  namespace instead of the bound GT owner ID.

## Ruled Out

- The current frozen rule set cannot adjudicate the C cohort or authorize the
  repair experiment. This is a control failure, not evidence that C owners lack
  visual or grounding support.
- The B2 example cannot be reported as target-basin suppression by a covered
  same-description owner under the unit's predeclared matched-control
  criterion. The registered geometry foils are diagnostics; they do not by
  themselves replace the required physical matched controls.
- Exact all-vocabulary FP32 KV-cache parity at `atol=1e-6`, `rtol=1e-5` is not
  available in the observed runtime. The decision-bearing scores remain the
  uncached reference.

## Unresolved

- How a successor should define owner-level peak shape without letting declared
  extent-submode labels split one high-overlap spatial mode. Merely whitelisting
  `part_or_whole_lobes` would preserve the defect rather than repair it.
- Whether collision should be defined by raw target-height change, a specified
  foil contrast, or a multivariate matched-foil rule. The current unit required
  agreement across the matched set and therefore held.
- How to recalibrate peak and prominence thresholds without pooling unlike root
  and B2 contexts into a degenerate minimum threshold. The held calibration was
  not authorized for C adjudication.
- The prevalence and identity of B1, B2, and high-confidence conditioned-C
  owners across the twelve images.
- Prevent-skip versus backfill route value, suffix retention, owner exchange,
  and duplicate chronology.
- Whether the new opt-in behavior-level KV-cache admission can reproduce the
  uncached basin conclusion. It remains an efficiency probe and cannot become
  decision-bearing without a prospectively revised owning contract.

## Not Claimed

- No C-sentinel outcome was opened, and no C owner was assigned.
- No twelve-image C prevalence estimate is reported.
- No crop-rescale, visual-resolution, architecture, training, or checkpoint
  recommendation follows from this held unit.
- No free-search candidate is promoted to a physical-owner basin because those
  candidates were not reviewed.
- No normalized basin-mass comparison is used where proposal measures were not
  comparable.

## Next Decision

A successor unit must be prospectively specified rather than reopening this
held run. The smallest useful successor would keep the same immutable raw
control scores, replace the label-first shape partition with an owner-level
spatial criterion calibrated separately by context, register the required B2
physical controls, and freeze one collision contrast before reading any C
outcomes. Only then should the twelve-image census or route-repair ladder
resume.
