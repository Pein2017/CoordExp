# Qwen3 Vision-Language Dense Enumeration Bottleneck

This investigation asks why a geometry-sorted Qwen3 Vision-Language
(`Qwen3-VL`) detector can describe and localize many individual Common Objects
in Context 80-category (`COCO-80`) objects, yet becomes conservative and
unstable when it must enumerate dense scenes in one autoregressive rollout.

The target remains diagnosis, not a final architecture. The completed program
has progressed from input-level spatial-policy controls through fixed-prefix
mode fragmentation, phrase-geometry transition state, fixed-encoding
post-vision routing, query-phase decomposition, and one bounded late pre-`x1`
geometry-owner basin switch. The newest unit adds strong case-level
`x1`-to-`x2` geometry transport, a fork-part late-extent shortcut, and
cross-row transition sensitivity without proving a physical owner or correct
covered-set update. It has not established autonomous selection, order-free
commit or coverage, clean-state synthesis, or a stable training intervention.
The weekly report and research compass own the current synthesis and next-
discriminator boundary.

## Reading Path

Start with the [weekly integrated research report for 2026-07-13 through
2026-07-16](2026-07-13-to-2026-07-16-weekly-research-report.md), then use the
durable program reading path below.

1. [Program research compass](compass.md)
2. [Overview and detailed hypothesis map](overview.md)
3. [Experiment units](experiments/)
4. [Completed masked spatial-policy and accepted-row prefix-policy unit](experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md)
5. [Executed results and bounded verdict](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md)
6. [Sampled-rescue transition unit](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/unit.md)
7. [Sampled-rescue transition results and bounded verdict](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md)
8. [Prefix-state phrase-geometry factorial](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/unit.md)
9. [Prefix-state phrase-geometry results and bounded verdict](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md)
10. [Native coherent-row commit-to-uncovered redistribution factorial](experiments/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/unit.md)
11. [Native coherent-row factorial results and execution-invariance verdict](experiments/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/results.md)
12. [Mixed-length, homogeneous, and equal-length batch coordinate-logit invariance probe](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/unit.md)
13. [Batch coordinate-logit invariance results and precision verdict](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md)
14. [Single-target visual-feature replay into homogeneous batch four](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/unit.md)
15. [Single-target visual-feature replay results and post-vision verdict](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/results.md)
16. [Selected-transition batch-precision prevalence screen](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/unit.md)
17. [Selected-transition batch-precision results and materiality verdict](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md)
18. [Repeated first-differing-slot full-coordinate-logit panel](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/unit.md)
19. [Repeated full-coordinate-logit results and numerical-branch verdict](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/results.md)
20. [Fixed-encoding object-centered spatial-eligibility crossover](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/unit.md)
21. [Fixed-encoding spatial-eligibility results and bounded verdict](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md)
22. [Fixed-encoding soft spatial-key bias dose response](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/unit.md)
23. [Fixed-encoding soft spatial-key bias results and closure verdict](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md)
24. [Fixed-encoding row-scoring-query-only spatial-key eligibility crossover](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/unit.md)
25. [Fixed-encoding row-scoring-query-only results and earlier-query-dependence verdict](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/results.md)
26. [Fixed-encoding earlier-query-only spatial-key eligibility factorial](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/unit.md)
27. [Fixed-encoding earlier-query-only factorial results and phase-separated interaction verdict](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/results.md)
28. [Fixed-encoding cross-region earlier-query and row-scoring-query spatial-key eligibility hybrid](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/unit.md)
29. [Cross-region hybrid results and asymmetric phrase-geometry verdict](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/results.md)
30. [Fixed-encoding count-balanced soft cross-region earlier-query and row-scoring-query spatial bias](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/unit.md)
31. [Count-balanced soft cross-region results and no-adjudication closure](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/results.md)
32. [Fixed-encoding conditional downstream layer-output residual-state portability gate](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/unit.md)
33. [Conditional downstream portability results and geometry-donor eligibility closure](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/results.md)
34. [Fixed-encoding persistent hard-routing geometry-donor eligibility screen](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/unit.md)
35. [Geometry-donor eligibility results and bounded successor decision](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md)
36. [Fixed-encoding persistent hard-routing geometry-state portability on image 7818](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/unit.md)
37. [Image-7818 geometry-state portability results and one-way `x1` basin verdict](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/results.md)
38. [Pre-execution readiness amendment](experiments/2026-07-13-spatial-scope-history-disentanglement/readiness-amendment.md)
39. [Pre-execution independent-review synthesis and revision gate](experiments/2026-07-13-spatial-scope-history-disentanglement/review.md)
40. [Human-audited rare-object trajectory genealogy and causal branch replay](experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/unit.md)
41. [Human-audited rare-object manual-review results and coordinate-coherence redirection](experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/results.md)
42. [Fixed-prefix complete-box coherence and progressive coordinate-release factorial](experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/unit.md)
43. [Fixed-prefix complete-box results and bounded coordinate-transport verdict](experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/results.md)

The review and readiness amendment preserve the pre-execution gate. The
completed unit and results record own current lifecycle and evidence status.

## Upstream Evidence

- [Autoregressive binding template study](../autoregressive-binding-template-study/)
- [Painted Ground-Truth transcription probe](../../ideas/qwen3-vl-painted-gt-transcription-probe/)
- [Separate selection, transcription, commit, and stop](../../decisions/separate-selection-transcription-commit-and-stop.md)
- [Require target-specific causal consumption](../../decisions/require-target-specific-causal-consumption.md)
- [Let architecture emerge from hypothesis gates](../../decisions/let-architecture-emerge-from-hypothesis-gates.md)
- [Use bagging as an object-support probe](../../decisions/use-bagging-as-an-object-support-probe.md)

## Authority

Everything in this investigation is non-normative research. Runtime artifacts
remain under `outputs/research/`; reusable implementation contracts, if later
justified, require a separate OpenSpec change.
