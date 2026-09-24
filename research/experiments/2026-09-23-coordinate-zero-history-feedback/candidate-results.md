# Zero-history feedback: CPU admission HOLD

Status: worker CPU candidate, lead acceptance pending. The frozen same-row x2-zero versus y2-zero control has **no eligible historical row** at any of the four target queries. No model was loaded, no logit was read, and no intervention effect was estimated. The exact [admission record](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-zero-history-feedback/selection/admission-v1.json) binds the accepted source trace, predecessor plan/acceptance, every target prefix, the existing source-logit cell and the complete prior-row counts.

| Frozen query | Prior complete rows | Prior x2=0 | Prior y2=0 | Same-row x2=y2=0 | Same-row x2=y1=0 |
|---|---:|---:|---:|---:|---:|
| 885 x2, relative +4, token 276 | 30 | 4 | 0 | 0 | 4 |
| 885 x2, relative +8, token 312 | 34 | 8 | 0 | 0 | 8 |
| 5586 x2, relative +2, token 357 | 39 | 2 | 0 | 0 | 2 |
| 5586 x2, relative +3, token 366 | 40 | 3 | 0 | 0 | 3 |

Every one of the seven frozen healthy traces has zero historical x2-zero complete rows, so none can furnish the stipulated control query. The zero y2 count is not a parser omission: the cited x2-zero rows are complete literal boxes with y2 values such as 38 or 47. The current x1 remains zero in all four target prefixes. The predecessor's pre-outcome best-legal replacement is 47 for both first-illegal cases, but neither the 1 nor 47 intervention can be admitted under the frozen matched-row rule.

The smallest alternate contrast would change *historical y1-zero* tokens in the same x2-zero rows as the wrong-axis/any-zero control. It would preserve edited row sets, counts, token-zero status, length and current-row context. It would **change the frozen y2 comparator**, and the seven frozen healthy traces still would not supply x2-zero histories. This is a proposal for a separate lead decision, not a substitution made here. Another control population would also require a new freeze. The present CPU finding says nothing about whether earlier x2-zero content causally affects later zero decisions.

The focused actual-source prefix test confirms all four query SHA256 values, causal x2 positions and zero current x1; a synthetic same-row counterexample rejects treating x2=0 in one row and y2=0 in another, or the current row, as a historical pair. Two tests pass. Four current/retained source pairs match; research knowledge, output layout and diff checks pass. The predecessor is untouched. There are zero new forwards, zero allocated GPU-seconds, no producer PID and no active job. No model-entry clock began in this package; the shared overnight GPU ledger remains at zero. The lead can carry its own common wall start forward without a reset.

Reproduce with `python -B -m pytest -q probes/training_set_completion/coordinate_order_knowledge/test_zero_history.py` and `python -B -m probes.training_set_completion.coordinate_order_knowledge.zero_history`. The candidate manifest binds the exact current code, capture, admission, report and tests. No Stage 2 continuation or successor was launched.
