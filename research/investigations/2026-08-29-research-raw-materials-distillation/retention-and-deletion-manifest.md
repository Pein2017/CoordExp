---
title: Exact raw-output retention and deletion manifest
type: investigation
role: deletion-manifest
authority: non_normative_research
status: executed
updated: 2026-08-29
---

# Exact retention and deletion manifest

Live snapshot: `2026-08-29T11:32:26Z`. Root: `/data/CoordExp/outputs/research`.

Only paths in `deletion-targets.txt` were eligible for deletion. The file is an exact path list; no glob or recursive parent outside the listed paths was authorized. `lsof` and GPU checks passed immediately before mutation; no listed path had an open handle.

## Rules

- `KEEP-IMAGE2299`: any qwen root whose path or descendant explicitly contains `image2299`; this conservative rule implements the user exception.
- `HOLD-ACTIVE`: qwen roots with live process/file handles at snapshot time.
- `KEEP-DEPENDENCY`: the eight-coordinate closeout, retained because Image2299 explicitly consumes `step-2444`.
- `DISTILL-DELETE`: canonical Markdown synthesis/records already preserve the decision-bearing result.

| Path | Size | Class | Reason |
|---|---:|---|---|
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement` | 12.17 GiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-14-sampled-rescue-object-transition-causal-replay` | 145.04 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias` | 501.24 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid` | 416.33 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate` | 730.49 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial` | 246.98 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover` | 10.15 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen` | 4.99 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover` | 1.01 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response` | 564.69 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance` | 21.66 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution` | 13.18 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit` | 1.70 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-prefix-state-phrase-geometry-factorial` | 52.83 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel` | 4.80 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-selected-transition-batch-precision-prevalence-screen` | 735.41 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four` | 1.11 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-15-visual-support-counterfactual-commit` | 3.27 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818` | 492.10 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial` | 51.34 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-16-human-audited-rare-object-trajectory-genealogy` | 1.21 MiB | HOLD-ACTIVE | live lsof handle |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-native-sibling-row-branch-value-and-commit-crossover` | 446.44 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-next-row-probability-transition-and-causal-source-trace` | 129.31 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon` | 701.89 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication` | 4.98 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen` | 4.48 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-human-resolved-dense-branch-value-and-calibration-screen` | 9.47 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-image2299-near-complete-human-relabel-successor-transition` | 25.96 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison` | 23.18 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout` | 18.19 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-common-object-prefix-permutation-short-horizon` | 84.28 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-complete-candidate-row-score-decomposition` | 606.45 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-local-branch-causality-and-downstream-coverage-value` | 26.59 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-same-covered-set-prefix-order-equivalence` | 4.53 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-sampled-history-target-reachability-and-complete-row-value` | 15.40 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen` | 25.01 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen` | 1.50 GiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-256-image-coordinate-boundary-training-screen` | 821.31 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen` | 712.77 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-earliest-shared-prefix-branch-and-trajectory-treatment` | 972.65 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-earliest-shared-prefix-branch-pilot` | 7.44 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-exact-greedy-terminal-rescue-training-screen` | 83.29 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-greedy-prefix-forced-owner-path-intervention` | 12.82 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-individual-trajectory-versus-union-support-audit` | 835.08 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction` | 645.26 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen` | 13.25 GiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-human-refined-greedy-set-completion-conditions` | 58.51 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-physical-owner-duplication-causality-and-training-treatment` | 2.35 GiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen` | 1.15 GiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-adjudication-salvage-gate` | 24.69 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-admission-census` | 155.27 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training` | 5.42 GiB | HOLD-ACTIVE | live lsof handle |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality` | 64.82 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-existing-checkpoint-transition-mechanism-decomposition` | 187.48 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-paired-natural-terminal-forced-opener-release` | 33.23 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-untouched-terminal-boundary-statistical-analysis` | 767.37 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-26-source-versus-transition-step36-forced-opener-owner-selection` | 28.49 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084` | 858.28 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-30-three-checkpoint-absolute-coordinate-confidence-visualization-v1` | 15.92 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-cohorts-v2-foil-sealed-final` | 12.09 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-cohorts-v2-foil-sealed-final-v2` | 12.09 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-landscape-and-repair` | 15.11 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task0-v2` | 18.55 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral` | 20.84 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final` | 21.00 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-fn-mechanism-decomposition-smoke-v1` | 105.44 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v7` | 5.71 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-scores-uncached-merged-v1` | 194.54 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-summary-uncached-reviewed-v4` | 11.39 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-all-person-owner-relative-route-landscape` | 22.83 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-crossing-boundary-owner-release-realization` | 25.10 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-full-canvas-visual-token-budget-intervention` | 1015.96 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-owner-accessibility-phenotype-census` | 2.89 GiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence` | 5.24 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-random-image2299-matched-mechanism-contrast` | 68.33 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control` | 2.36 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification` | 11.67 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-image2299-prospective-mechanism-extension` | 144.82 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission` | 69.49 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover` | 134.19 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication` | 532.34 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-07-s-k10-h20-natural-crossover` | 25.96 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-human13-k-union-to-greedy-overfit-screen` | 2.10 GiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe` | 10.46 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-missing-arms-successor` | 855.05 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-on-policy-first-bottleneck-successor` | 42.90 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-row-contrast-geometry-preservation-successor` | 387.62 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-14-human13-k-trajectory-rp-crossover-screen` | 44.18 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical` | 1.07 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-all-hf-shared-surface-trajectory-credit-vertical` | 3.08 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe` | 444.27 KiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-22-human13-owner-credit-nk-factorial` | 2.75 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-23-human13-n13-k4k8-corrected-geometry-probe` | 19.25 MiB | DISTILL-DELETE | covered by dense-enumeration synthesis |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-24-image2299-decision-distribution-microscope` | 335.37 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-24-image2299-near-policy-prefix-dose-response` | 19.79 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-24-image2299-second-row-prefix-spatial-counterfactual` | 33.37 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-geo-sorted-xy-step2444-probes` | 3.37 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-natural-and-gt-prefix-free-decode` | 1.88 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-natural-history-gt-person-prefix-free-decode` | 1.26 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-safe-successor-rectangle-guard-training-vertical` | 462.69 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-safe-successor-self-prefix-training-vertical` | 77.08 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-xy-adapter-embedding-composition` | 309.84 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-xy-gt23-relative-barrier-training` | 154.21 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-xy-prefix-safe-deficit-compilation` | 2.18 GiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-xy-same-owner-serialization-sentinel` | 491.56 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-xy-single-edge-owner-compilation` | 760.65 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-25-image2299-xy-step1-delta-backtracking` | 155.76 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-26-image2299-full-root-detached-margin` | 323.30 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-26-image2299-onpolicy-self-prefix-insertion` | 4.62 GiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-26-image2299-set-level-compilation` | 211.06 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-anchor-donor-portability-census` | 6.53 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-complete-row-margin-compilation` | 2.03 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-full-trajectory-vertical` | 21.12 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-gt32-cross-prefix-radius-refinement` | 353.31 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-gt32-cross-prefix-retention-projection` | 607.09 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-gt32-successor-restoration` | 1.84 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-gt32-two-boundary-compilation` | 1.86 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-k24-owner-set-score-function-compatibility` | 4.13 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-multitemperature-full-trajectory` | 10.31 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-native-guarded-unmatched-row-cleanup` | 989.62 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-native-state-successor-lattice` | 1.63 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-output-only-active-competitor-closure` | 4.37 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-owner-aware-sparse-margin` | 183.65 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-owner-level-near-miss-line-search` | 873.61 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-owner-set-preserving-gradient-feasibility` | 2.38 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-parallel-owner-guard-tuning` | 11.80 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-theta-b-affine-union-compilation` | 247.03 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-theta-b-forced-gt19-gt34-union` | 403.43 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-theta-b-output-only-tied-row-discriminator` | 267.03 KiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-27-image2299-tied-embedding-balance-probe` | 9.57 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-28-image2299-fixed-natural-support-panel` | 148.74 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-28-image2299-parent-a-k24-structured-event-ce` | 161.66 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-28-image2299-row2-owner16-owner29-sibling-audit` | 57.66 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-29-image2299-exact-prefix-first-step-scale-window` | 618.38 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-29-image2299-exact-prefix-small-step-online50` | 110.39 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-29-image2299-parent-a-k24-exact-prefix-owner-ce` | 79.32 MiB | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/logs` | 751.00 B | KEEP-IMAGE2299 | Image2299 appears in bound output path |
| `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge` | 13.79 GiB | DISTILL-DELETE | canonical summary: pvci-causal-proposal-bridge.md |
| `/data/CoordExp/outputs/research/qwen3-vl-native-text-coordinate-val200` | 1.52 GiB | DISTILL-DELETE | canonical summary: native-text-coordinate-val200.md |
| `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation` | 926.46 MiB | DISTILL-DELETE | canonical summary: pi-lightweight-worker-ablation.md |
| `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision` | 779.91 MiB | KEEP-DEPENDENCY | Image2299 consumes closeout/checkpoints/step-2444 |

Counts: `141` roots inventoried; `84` exact deletion targets; `57` retained/held roots.

Estimated eligible bytes: **59.44 GiB**.

The qwen classification is root-level. A root was never deleted if an Image2299 descendant was present, even if most bytes are older historical material. This is intentionally conservative and can be narrowed later with a new manifest.

Deletion receipt: 84 exact roots were removed with `/usr/bin/find <target> -xdev -depth -delete`. The manifest's eligible file payload was 59.44 GiB. After deletion, `outputs/research` occupies 21,001,461,760 bytes (about 19.56 GiB); the remaining qwen family occupies 20,182,781,952 bytes and contains 56 top-level roots. Active handles remain only under the two `HOLD-ACTIVE` roots.
