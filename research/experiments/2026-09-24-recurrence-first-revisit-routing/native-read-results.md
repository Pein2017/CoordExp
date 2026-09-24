# Native historical reading: first revisit and successful progress

Lead-accepted bounded results, 2026-09-24. The [first-revisit acceptance](lead-native-read-first-case-acceptance.json) covers three full-prefix model calls; the [bowl-progress acceptance](lead-bowl-progress-cpu-acceptance-v1.json) covers only three finalized records recovered through a separately admitted CPU readback. The seven-cell GPU attempt remains terminal technical-invalid. Its unfinalized earlier-row cell remains unanswered, and the cow cells remain unrun. This finite panel is closed at the no-retry boundary, without completing all planned controls.

## Finding

The latest completed row does not exert a uniformly repeat-promoting influence on these native coordinate choices. At the first revisit, masking it has an extent-dependent effect. At the following successful transition, masking it strongly reduces the next bowl's x1 relative to both prior extents. That is a counterexample to treating historical-row reading as uniformly harmful to progress; it is not yet a physical-owner outcome under intervention.

The original four-request fresh-18 batch contains target train313465 at index3. Native rows0/1 localize the same foreground bowl A with different extents; row2 localizes a distinct upper bowl B. Fixed x1 tokens are A0=151670 (0), A1=151675 (5), B=151827 (157). Each intervention blocks only current-row-prefix access to the latest completed row, across all heads and28 text layers, while retaining native tokens, positions, images, companions and historical states. Current-row Q/K/V can respond downstream.

| Native state | Masked history | Change in z(B)-z(A1) | Change in z(B)-z(A0) | B rank: native → masked | Global winner: native → masked |
| --- | --- | ---: | ---: | --- | --- |
| Row1, first physical A revisit | Row0 A0 | +0.645916 nat | -0.855429 nat | 203 →162 | 151675 →151670 |
| Row2, successful B arrival | Row1 A1 | -9.976613 nat | -7.387984 nat | 1 →192 | 151827 →151671 |

All changes are masked minus native; the numerical interpretive deadband remains0.01nat. At row2, B-A1 changes from+3.251831 to-6.724782, and B-A0 from+0.604050 to-6.783935. B probability falls from0.0610505 to0.000145367; full-vocabulary TV is0.847765. The third winning token151671 is not a completed box or an owner label. Row2-minus-row1 changes are-10.622529nat primary and-6.532556nat secondary. The secondary sign stays negative in both states; the primary sign reverses. No coordinate-only renormalization or total-owner-mass interpretation is used.

## Mechanistic implication and alternatives

A useful working hypothesis is that prior detections provide a contextual reference that can support both localization and movement to the next region. In the first state, the historical read differentiates alternative extents around A; in the second, its availability supports the native B coordinate against those alternatives. This hypothesis goes beyond a uniform copying account, but does not establish what semantic information the model extracts from the row.

The strongest unresolved alternative is a broader change in current-row computation when an entire attention source is removed. Softmax redistribution and changed current queries can alter the coordinate distribution without an explicit visited-owner representation. The two states also differ in history content, length and position. Their effect difference cannot isolate coveredness, attribute the failure to K alone, or identify a unique circuit. The missing earlier-row and cow controls prevent recency or cross-image specificity claims.

The next decision-changing gap is the link to complete physical detections: a changed x1 winner may produce a valid new box, another extent of A, or invalid geometry. A separately frozen, bounded completion comparison would test that link before further layer/key localization. Distinguishing a visited-owner signal from generic localization context would additionally need a content-selective, matched intervention. Neither is launched or admitted here.

## Evidence, failure and cost

The [worker CPU candidate](native-read-bowl-progress-cpu-candidate.md) and [versioned readback](supporting/native-read-bowl-progress-cpu-readback-v1.json) bind source images/receipts, full-batch geometry, observed consumers and three raw tensors. Lead replayed the CPU reader to a fresh temporary output and reproduced its JSON/hash exactly, then independently recomputed NumPy-reported quantities with PyTorch FP64 (maximum discrepancy1.78e-15). Active source trace parity maximum error was3.0518e-5 under2e-4. Sham vectors, all-prior target layer states and companion layer states/vectors match exactly. Only the50 specified mask entries change; all28 actual attention entries are recorded. [Independent verification](supporting/lead-bowl-progress-cpu-verification-v1.json) retains the checks.

The [failure ruling](lead-native-read-failure-ruling-v1.json) preserves the bad cross-rectangle complement comparator and the missing fourth consumer receipt; it does not synthesize evidence or relax the mask gate. Across the accepted first case and failed attempt, seven model/seven vision calls were charged, with zero generated tokens. Panel charge remains41.101082854 GPU seconds and sequence charge0.249602809909 GPU-hours of8. CPU recovery and lead verification added zero model calls and GPU time. No retry, cow launch, or successor is authorized.
