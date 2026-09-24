# First arrivals: finite ranking and conditional coordinate realization

**Lead-accepted; finite package closed.** The
[Stage2 acceptance receipt](lead-stage2-acceptance-v1.json) owns the final
27-cell accounting: 24 new cells and three reused from the separately
[accepted bowl qualification](lead-stage1-acceptance-v1.json). Five families,
seven native landmarks and ten fixed A/N pairs were analyzed. The
[state](state.json) owns lifecycle; there is no active model job or successor.

## What this test established

Every tested complete native row A outranks its fixed canonical new-owner row N
outside the numerical deadband. There are **zero N-over-A complete-row mismatch
witnesses among ten finite comparisons**. This narrows the proposed local greedy
versus complete-row discrepancy on these rows; it does not estimate total owner
probability, identify an internally selected object, or rule out other boxes.

Supplying N's x1 can nevertheless produce the remaining three coordinates of N.
The bowl does so at both first and second arrival; the glass and both cow
control branches do so too. This conditional realization is available outside
the recurrent bowl case. Normal here means nonreturn of the selected target,
not a globally healthy scene or a matched factorial control.

| Case | Accepted observation |
| --- | --- |
| Bowl, first and second arrival | Both A-favoring comparisons yield the same supplied-x1 N box `[161,0,415,241]`, IoU 0.8133. Native reaches N at its third and next row respectively. |
| Invalid multi-chair proposal, first arrival row 4 | Either fixed N x1 produces its corresponding foreground chair, with IoU 0.7124 or 0.8950. |
| Same invalid proposal, second arrival row 5 | The same x1 values instead produce a stage-chair region or a broad multi-object box. Neither is the fixed N. |
| Target-nonreturn controls | Glass and both cow branches realize N. The book-area box has IoU 0.4605 and partial extent; retain physical identity UNKNOWN, not a false-detection label. |

The native broad chair row spans multiple owners even though its coordinates
are geometrically valid. Its two arrivals are an invalid-proposal recurrence,
not a valid single-owner revisit. Both recurrent families naturally leave these
short episodes; no persistent-burst rescue was established.

## The informative adjacent-state contrast

The chair A row is exactly `[0,692,999,999]` at both native landmarks. Its complete
log probability falls from −12.0108 to −14.1826: the second is about **0.114 times**
the first, while greedy still emits the same row. This is a counterexample to a
universal claim that each repetition must increase the repeated row's absolute
probability. It does not reject all history-feedback mechanisms: greedy choice
depends on coordinate-wise ranking, not monotonic whole-row probability.

The supplied x1 response changes immediately at the first free coordinate:

| Supplied x1 | Free y1 at row 4 | Free y1 at row 5 |
| ---: | ---: | ---: |
| 510 | 768 | 413 |
| 418 | 873 | 0 |

The image, model, candidate and supplied x1 stay fixed, while the native history
grows by one nine-token row. Thus the conditional mapping from x1 to the rest of
the box changes between these histories. Added row content, contextual state and
position advance together. Their separate causal contributions are unidentified;
x1 is not demonstrated to be a stable address for an object across histories.

## Acceptance and limits

The lead freshly reran CPU cold readback, recomputed all new score sums and D
values, checked all nine sham/N input differences and masks, recalculated proxy
IoUs, checked terminality and visually reviewed the outcomes. One bounded
independent caller/provenance review found no decision-changing semantic defect.
Every new landmark passed native trace and complete-A-score parity, with maximum
error 6.2943e-5 against the 2e-4 tolerance; separate A shams match native tokens.
Consumer mutations test verifier sensitivity, not extra modified-model ablations.

Source provenance is explicitly incomplete: four direct imports were captured
after execution. `prepare.py` has an earlier CPU-receipt hash; `artifacts.py`,
`input_identity.py` and `runtime_loading.py` lack prelaunch byte hashes. Their
mtimes do not prove launch-time bytes. The actual-input, effective-model and
per-landmark replay/score evidence supports this finite execution acceptance;
**complete prelaunch source freeze is not claimed**. The original records remain
unchanged, and no replay was launched solely to fill this archival gap.

The admitted set contains only one first-valid recurrent family and one
first-invalid multi-owner family, with no admitted third arrival. Thirteen
families across twelve images were reviewed from 273 source images; all other
owner, invalidity and candidate HOLDs remain visible. These are descriptive
cases, not population rates or a unified explanation of repetition.

All setup, failed attempts and diagnostics are charged in the linked ledger:
**0.0579751 allocated GPU-hours** of the eight-hour package ceiling. All jobs are
terminal. The frozen question is answered on its finite support, so this screen
closes without more candidates or samples.

If a new investigation is opened, the concrete remaining discriminator is
whether the chair y1 change follows the added row's content at fixed positions
or position advancement with content held fixed. It must predeclare its outcome,
preserve native/sham behavior and be compared with the earlier content/phase
results. This is a proposed causal follow-up, not a mechanism finding or launch
authorization.
