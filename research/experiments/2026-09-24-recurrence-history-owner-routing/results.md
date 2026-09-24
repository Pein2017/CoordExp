# Historical coordinates change routing, but the two-bowl predictions fail

Lead-accepted and closed, 2026-09-24. [Acceptance](lead-acceptance-v1.json) binds the [candidate](candidate-results.md), [independent verification](supporting/lead-verification-v1.json), full raw vectors and visual rulings. The original source, four-request batch, current row1 position, native mask and supplied current bowl header are fixed. Only the preceding row's four coordinate tokens change; each arm then greedily completes the current row.

| Earlier coordinates | Complete current box | Lead physical readout |
| --- | --- | --- |
| Original A0 | [5,197,217,407] | Inner bowl A1 |
| Identity-write sham A0 | same | Same A1 |
| Definite A1 donor | [0,157,277,502] | Broad pot/inner-bowl extent, UNKNOWN/HOLD |
| Definite upper bowl B donor | [177,436,460,828] | Other visible object: central citrus juicer/container assembly |

All four rows have canonical syntax and positive-area geometry. The B-donor [overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-history-owner-routing/attempt-001/overlay-B-donor.png) clearly identifies a region different from both reference bowls. Its label bowl may reflect object granularity or class-space mismatch; absence of a matching annotation does not make the object hallucinated. The A1-donor [overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-history-owner-routing/attempt-001/overlay-A1-donor.png) retains the user's pot/bowl ambiguity and is not forced into A1 despite IoU0.503720.

The strict unchanged-tail prediction fails for both donors, showing that a pure content-insensitive row-count/position description is insufficient for these outputs. Both predeclared two-bowl pairs also fail: B-donor produces neither A1 (required by the exclusion pair) nor B (required by the anchoring pair). **This rejects the restricted two-bowl pair, not generic visited-owner exclusion:** a third object may itself be a new owner. The A1-donor ambiguity stays in the denominator, not relabeled as a success or a physical false positive. No literal copying or semantic owner ledger follows.

A material conditioning limit remains. The current bowl header was supplied from original row1 even when its history was changed to B. In the full original source, the row after B is scissors, not another bowl; B-donor's juicer output is therefore not evidence about unconstrained next-category choice. A separate free-header contrast can test this gap. A working spatial/order hypothesis predicts that B-only history will naturally choose scissors as in B's native successor, while a naturally continued bowl branch predicts the same bowl/juicer outcome without header supply. Neither forecast is established here.

Lead independently reloaded all20 raw vocabulary tensors, recomputed greedy choices, reconstructed original batch plus exact four-token donor and own generated history, and verified every actual input/native-mask binding and28-layer entry. Native and sham vectors exactly reproduce accepted references; all companion vectors/states are unchanged. Native source max error2.8611e-5 is below2e-4; independent NumPy/PyTorch logP differences are below1.78e-15. Donor target historical states were correctly allowed to change.

Exactly20 model/20 vision forwards and20 emitted tokens completed; sole model job exited0 and PID1812015 is absent. Parent outer charge57.915783612seconds; cumulative sequence0.291357630402GPU-hours. The disclosed log-directory error preceded child spawn and was a routine setup correction with no model call, not a model retry or scientific invalid result. No further arm or successor runs under this closed unit.
