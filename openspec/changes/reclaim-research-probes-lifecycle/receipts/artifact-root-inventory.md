# Artifact-root inventory (task 6.4, report-only)

Audited at HEAD `9f06b47889c8e815e4da5a2c0a33f7d468a71cc5`, 2026-08-28.
**Nothing under `outputs/` was deleted, moved, or modified to produce this
receipt.** All commands were read-only (`du`, `find`, `grep`), each under
`timeout 120` (`timeout 180` for the one large recursive `find -printf`
pass, which completed in under a second on a warm cache; no command timed
out).

## Method

For each artifact root, `du -h -d 1` gave size, a single recursive
`find -mindepth 2 -type f -printf '%h\t%TY-%Tm-%Td\n'` reduced to newest
mtime and file count per top-level subdir (one pass over the whole
65G/105K-file `qwen3-vl-dense-enumeration` tree rather than 170 separate
`find` calls, for speed), and binding was
`grep -rl -F '<full artifact path>' research memories docs/history
openspec/changes`. Classification:

- **BOUND-ACTIVE**: cited by `memories/current.md`, or by a `research/` unit
  (outside `research/archive/`) whose frontmatter `status:` is not
  complete/closed (e.g. `active`).
- **BOUND-COMPLETE**: cited only by `research/` unit(s) (outside
  `research/archive/`) whose frontmatter `status:` is complete/closed
  (`complete`, `completed`, `closed`, `retired`, `superseded`, `redirected`,
  or a `complete_*` variant).
- **BOUND-ARCHIVED**: cited only under `research/archive/**`,
  `docs/history/**`, or `openspec/changes/archive/**`.
- **UNBOUND**: no hit anywhere in that search.

**Important caveat found during this audit**: the live, not-yet-returned
direction worktree `/data/CoordExp/.worktrees/image2299-mechanism-microscope`
keeps its own `research/` tree with its own experiment units, which this
change's search domain (`research memories docs/history openspec/changes` in
*this* worktree) cannot see. A full-path re-check against that worktree's
`research/` and `memories/` found 33 of the 104 nominally-UNBOUND
`qwen3-vl-dense-enumeration` subdirectories (5.20 GB) are in fact cited by
path from an image2299 unit — these are **not** orphaned, just not yet
returned. A further name-only check (same-named experiment directory exists,
even without a literal path citation — the normal 1:1 naming convention
between an experiment unit and its artifact root) found 26 more
research-probes-side and 37 more image2299-side matches with no literal path
citation (1.35 GB and 10.57 GB respectively) — plausibly bound by convention
but not provably cited; these are flagged separately from the 41
directories / 3.53 GB that have neither a path citation nor a same-named unit
anywhere and are the actual highest-confidence UNBOUND set. The table below
reports the strict path-citation class per the task's own definition, with a
`note` column carrying this refinement.

## Table 1 — top-level roots under `/data/CoordExp/outputs/research/` (82G total on disk; 68.9G/65GiB of it is one subtree, broken out in Table 2)

| root | size | newest mtime | class | binding evidence |
|---|---:|---|---|---|
| `qwen3-vl-dense-enumeration/` | 65G | 2026-08-27 | MIXED — see Table 2 (46 BOUND-COMPLETE, 19 BOUND-ACTIVE, 1 BOUND-ARCHIVED, 104 UNBOUND by strict path citation) | n/a, rolled up |
| `pvci_causal_proposal_bridge/` | 14G | 2026-07-12 | BOUND-COMPLETE | `research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/unit.md` (`status: complete`) |
| `qwen3-vl-native-text-coordinate-val200/` | 1.6G | (not queried further; zero citers found) | UNBOUND | no hit in `research memories docs/history openspec/changes` |
| `pi-lightweight-worker-ablation/` | 877M | 2026-07-23 | BOUND-ARCHIVED | `research/archive/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/{unit,results}.md`, `.../index.md`; also named by 3 `memories/notes/` (not `current.md`-linked) |
| `eight-coordinate-bbox-supervision/` | 781M | 2026-08-05 | BOUND-ACTIVE | cited by `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-05-static-dynamic-owner-interface-crossover/unit.md` (`status: active`) and 3 complete units; mixed set -> ACTIVE |

Also present, out of scope for this table but confirmed:
- `/data/CoordExp/outputs/research-probe-infras/` — 3.7M, sealed CPU
  receipts. **KEEP** regardless of binding, per instructions.
- One symlink found under `outputs/research/`:
  `qwen3-vl-dense-enumeration/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/rescale_32_1024_bbox`
  -> `/data/CoordExp/public_data/coco/rescale_32_1024_bbox` (points outside
  `outputs/`, into the shared public-data root; not an artifact-to-artifact
  symlink).

## Table 2 — one level deeper: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/` (170 subdirs, 68.86G / 104,731 files)

Totals by class: BOUND-COMPLETE 34.54 GB (46 dirs) · BOUND-ACTIVE 18.74 GB
(19 dirs, includes the 2 dirs cited by `memories/current.md`) · UNBOUND
15.45 GB (104 dirs by strict path citation; of that, 5.20 GB / 33 dirs is
actually image2299-bound by path, 1.35 GB / 26 dirs + 10.57 GB / 37 dirs is
same-name-only bound, and 3.53 GB / 41 dirs has no citation and no
same-named unit anywhere) · BOUND-ARCHIVED 0.14 GB (1 dir).

| artifact root (under `.../qwen3-vl-dense-enumeration/`) | size (MB) | newest mtime | files | class | note |
|---|---:|---|---:|---|---|
| `2026-07-22-constant-dose-image-breadth-treatment-screen` | 14225.6 | 2026-07-23 | 4732 | BOUND-ACTIVE |  |
| `2026-07-13-spatial-scope-history-disentanglement` | 13138.8 | 2026-07-14 | 85749 | BOUND-COMPLETE |  |
| `2026-07-24-prefix-local-and-on-policy-owner-set-training` | 5817.1 | 2026-07-25 | 1448 | BOUND-COMPLETE |  |
| `2026-08-26-image2299-onpolicy-self-prefix-insertion` | 4960.3 | 2026-08-27 | 545 | UNBOUND | same-name unit dir exists in image2299 worktree (no path citation) |
| `2026-08-03-sorted-owner-accessibility-phenotype-census` | 3103.6 | 2026-08-03 | 302 | BOUND-COMPLETE |  |
| `2026-07-22-physical-owner-duplication-causality-and-training-treatment` | 2523.6 | 2026-07-22 | 737 | BOUND-COMPLETE |  |
| `2026-08-25-image2299-xy-prefix-safe-deficit-compilation` | 2343.5 | 2026-08-25 | 278 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-12-human13-k-union-to-greedy-overfit-screen` | 2255.7 | 2026-08-12 | 217 | BOUND-ACTIVE |  |
| `2026-07-30-iterative-forced-continue-exact-native` | 1776.1 | 2026-07-30 | 508 | UNBOUND | no same-name unit dir anywhere |
| `2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen` | 1615.5 | 2026-07-20 | 635 | BOUND-COMPLETE |  |
| `2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen` | 1236.8 | 2026-07-22 | 342 | BOUND-COMPLETE |  |
| `2026-08-03-sorted-full-canvas-visual-token-budget-intervention` | 1065.5 | 2026-08-03 | 164 | BOUND-COMPLETE |  |
| `2026-07-21-earliest-shared-prefix-branch-and-trajectory-treatment` | 1019.9 | 2026-07-21 | 11 | BOUND-COMPLETE |  |
| `2026-07-29-three-checkpoint-human-refined12-max3084` | 900.1 | 2026-07-29 | 202 | BOUND-ACTIVE |  |
| `2026-08-13-human13-missing-arms-successor` | 896.9 | 2026-08-13 | 106 | BOUND-COMPLETE |  |
| `2026-07-21-individual-trajectory-versus-union-support-audit` | 875.7 | 2026-07-21 | 49 | BOUND-COMPLETE |  |
| `2026-07-21-256-image-coordinate-boundary-training-screen` | 861.6 | 2026-07-21 | 380 | BOUND-COMPLETE |  |
| `2026-08-25-image2299-xy-single-edge-owner-compilation` | 798.3 | 2026-08-25 | 334 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-21-best-sampled-trajectory-positive-row-imitation-screen` | 747.8 | 2026-07-21 | 352 | BOUND-ACTIVE |  |
| `2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction` | 676.9 | 2026-07-21 | 238 | BOUND-COMPLETE |  |
| `2026-08-01-sorted-owner-basin-candidates-six-context-smoke-v2` | 603.7 | 2026-08-01 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-06-natural-boundary-routing-history-replication` | 558.5 | 2026-08-07 | 180 | BOUND-COMPLETE |  |
| `2026-08-01-sorted-owner-basin-task4-control-candidates-final` | 521.0 | 2026-08-01 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-25-image2299-safe-successor-rectangle-guard-training-vertical` | 485.3 | 2026-08-25 | 45 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-17-native-sibling-row-branch-value-and-commit-crossover` | 469.2 | 2026-07-17 | 1726 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-13-human13-row-contrast-geometry-preservation-successor` | 406.6 | 2026-08-13 | 60 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-24-image2299-decision-distribution-microscope` | 351.8 | 2026-08-24 | 63 | UNBOUND | same-name unit dir exists in image2299 worktree (no path citation) |
| `2026-08-26-image2299-full-root-detached-margin` | 339.1 | 2026-08-26 | 28 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-25-image2299-xy-adapter-embedding-composition` | 325.0 | 2026-08-25 | 30 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-26-image2299-set-level-compilation` | 221.4 | 2026-08-26 | 18 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-02-sorted-owner-basin-task4-control-scores-uncached-merged-v1` | 204.0 | 2026-08-02 | 2 | BOUND-COMPLETE |  |
| `2026-07-25-existing-checkpoint-transition-mechanism-decomposition` | 196.9 | 2026-07-25 | 349 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-27-image2299-owner-aware-sparse-margin` | 192.6 | 2026-08-27 | 185 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-29-iterative-forced-continue-extreme-capacity` | 178.8 | 2026-07-29 | 10 | UNBOUND | no same-name unit dir anywhere |
| `2026-07-23-trajectory-owner-set-admission-census` | 164.4 | 2026-07-23 | 115 | BOUND-ACTIVE |  |
| `2026-08-25-image2299-xy-step1-delta-backtracking` | 163.4 | 2026-08-25 | 29 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-25-image2299-xy-gt23-relative-barrier-training` | 161.8 | 2026-08-25 | 20 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-14-sampled-rescue-object-transition-causal-replay` | 153.1 | 2026-07-14 | 204 | BOUND-COMPLETE |  |
| `2026-08-04-sorted-image2299-prospective-mechanism-extension` | 151.9 | 2026-08-04 | 102 | BOUND-COMPLETE |  |
| `2026-08-05-static-dynamic-owner-interface-crossover` | 141.0 | 2026-08-06 | 274 | BOUND-ACTIVE |  |
| `2026-07-17-next-row-probability-transition-and-causal-source-trace` | 135.9 | 2026-07-17 | 419 | BOUND-ARCHIVED |  |
| `2026-08-02-sorted-fn-mechanism-decomposition-smoke-v1` | 110.6 | 2026-08-03 | 72 | BOUND-ACTIVE |  |
| `2026-07-19-common-object-prefix-permutation-short-horizon` | 88.5 | 2026-07-20 | 77 | BOUND-COMPLETE |  |
| `2026-07-21-exact-greedy-terminal-rescue-training-screen` | 87.5 | 2026-07-21 | 116 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-25-image2299-safe-successor-self-prefix-training-vertical` | 80.9 | 2026-08-25 | 16 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-02-sorted-owner-basin-task4-control-shard-gt7511-22-root-v1` | 72.6 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-04-random-image2299-matched-mechanism-contrast` | 71.7 | 2026-08-04 | 50 | BOUND-COMPLETE |  |
| `2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality` | 68.0 | 2026-07-25 | 27 | BOUND-COMPLETE |  |
| `2026-07-22-human-refined-greedy-set-completion-conditions` | 61.4 | 2026-07-22 | 39 | BOUND-ACTIVE |  |
| `2026-07-15-prefix-state-phrase-geometry-factorial` | 55.7 | 2026-07-15 | 73 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial` | 53.9 | 2026-07-16 | 38 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-01-sorted-owner-basin-task4-control-scores-uncached-timing-ctx7511-26-v1` | 47.8 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-shard-gt7511-26-root-v1` | 47.8 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-14-human13-k-trajectory-rp-crossover-screen` | 46.5 | 2026-08-15 | 48 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-02-sorted-owner-basin-task4-control-shard-b2-before-v1` | 45.7 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-shard-b2-after-v1` | 45.5 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-13-human13-on-policy-first-bottleneck-successor` | 45.4 | 2026-08-13 | 200 | BOUND-COMPLETE |  |
| `2026-07-30-four-checkpoint-human-refined12-max3084` | 41.2 | 2026-07-30 | 33 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-24-image2299-second-row-prefix-spatial-counterfactual` | 35.1 | 2026-08-24 | 377 | UNBOUND | same-name unit dir exists in image2299 worktree (no path citation) |
| `2026-07-25-paired-natural-terminal-forced-opener-release` | 34.9 | 2026-07-25 | 67 | BOUND-COMPLETE |  |
| `2026-07-26-source-versus-transition-step36-forced-opener-owner-selection` | 29.9 | 2026-07-26 | 56 | BOUND-COMPLETE |  |
| `2026-07-19-local-branch-causality-and-downstream-coverage-value` | 27.9 | 2026-07-19 | 11 | BOUND-COMPLETE |  |
| `2026-08-07-s-k10-h20-natural-crossover` | 27.3 | 2026-08-07 | 78 | BOUND-COMPLETE |  |
| `2026-07-18-image2299-near-complete-human-relabel-successor-transition` | 27.3 | 2026-07-18 | 111 | BOUND-COMPLETE |  |
| `2026-08-03-sorted-crossing-boundary-owner-release-realization` | 26.5 | 2026-08-04 | 124 | BOUND-COMPLETE |  |
| `2026-07-20-matched-random-sorted-prefix-order-screen` | 26.3 | 2026-07-20 | 25 | BOUND-COMPLETE |  |
| `2026-07-23-trajectory-owner-set-adjudication-salvage-gate` | 26.0 | 2026-07-23 | 64 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-18-matched-objective-coordinate-branch-signature-comparison` | 24.4 | 2026-07-18 | 42 | BOUND-COMPLETE |  |
| `2026-08-03-sorted-all-person-owner-relative-route-landscape` | 24.1 | 2026-08-03 | 70 | BOUND-ACTIVE |  |
| `2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance` | 22.7 | 2026-07-15 | 3 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-27-image2299-full-trajectory-vertical` | 22.2 | 2026-08-27 | 133 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final` | 22.0 | 2026-08-01 | 8 | BOUND-ACTIVE |  |
| `2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral` | 21.9 | 2026-08-01 | 8 | BOUND-ACTIVE |  |
| `2026-08-24-image2299-near-policy-prefix-dose-response` | 20.8 | 2026-08-24 | 92 | UNBOUND | same-name unit dir exists in image2299 worktree (no path citation) |
| `2026-08-23-human13-n13-k4k8-corrected-geometry-probe` | 20.3 | 2026-08-23 | 65 | BOUND-ACTIVE |  |
| `2026-08-01-sorted-owner-basin-task0-v2` | 19.5 | 2026-08-01 | 7 | BOUND-ACTIVE |  |
| `2026-07-18-person25-dominant-owner-commit-and-persistence-closeout` | 19.1 | 2026-07-18 | 24 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-30-three-checkpoint-coordinate-confidence-visualization-v1` | 16.7 | 2026-07-30 | 13 | UNBOUND | no same-name unit dir anywhere |
| `2026-07-30-three-checkpoint-absolute-coordinate-confidence-visualization-v1` | 16.7 | 2026-07-30 | 13 | BOUND-COMPLETE |  |
| `2026-07-19-sampled-history-target-reachability-and-complete-row-value` | 16.2 | 2026-07-19 | 26 | BOUND-COMPLETE |  |
| `2026-08-01-sorted-owner-basin-landscape-and-repair` | 15.8 | 2026-08-01 | 7 | BOUND-COMPLETE |  |
| `2026-07-15-native-coherent-row-commit-to-uncovered-redistribution` | 13.9 | 2026-07-15 | 63 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-21-greedy-prefix-forced-owner-path-intervention` | 13.5 | 2026-07-21 | 12 | BOUND-ACTIVE |  |
| `2026-08-01-sorted-owner-basin-cohorts-v2-foil-sealed-final-v2` | 12.7 | 2026-08-01 | 5 | BOUND-COMPLETE |  |
| `2026-08-01-sorted-owner-basin-cohorts-v2-foil-sealed-final` | 12.7 | 2026-08-01 | 5 | BOUND-COMPLETE |  |
| `2026-08-01-sorted-owner-basin-cohorts-v2-final` | 12.7 | 2026-08-01 | 5 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-27-image2299-parallel-owner-guard-tuning` | 12.4 | 2026-08-27 | 9 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification` | 12.3 | 2026-08-04 | 47 | BOUND-COMPLETE |  |
| `2026-08-02-sorted-owner-basin-task4-control-summary-uncached-reviewed-v4` | 11.9 | 2026-08-02 | 2 | BOUND-COMPLETE |  |
| `2026-08-12-owner-bridge-step611-recall-probe` | 11.0 | 2026-08-12 | 56 | BOUND-COMPLETE |  |
| `2026-08-27-image2299-multitemperature-full-trajectory` | 10.9 | 2026-08-27 | 10 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover` | 10.7 | 2026-07-15 | 12 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-27-image2299-tied-embedding-balance-probe` | 10.1 | 2026-08-27 | 8 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-18-human-resolved-dense-branch-value-and-calibration-screen` | 10.0 | 2026-07-18 | 38 | BOUND-ACTIVE |  |
| `2026-07-21-earliest-shared-prefix-branch-pilot` | 7.8 | 2026-07-21 | 3 | BOUND-COMPLETE |  |
| `2026-08-27-image2299-anchor-donor-portability-census` | 6.9 | 2026-08-27 | 9 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-02-sorted-owner-basin-task4-control-summary-uncached-reviewed-v3` | 6.7 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence` | 5.5 | 2026-08-03 | 22 | BOUND-COMPLETE |  |
| `2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication` | 5.3 | 2026-07-18 | 41 | BOUND-COMPLETE |  |
| `2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen` | 5.3 | 2026-07-15 | 7 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel` | 5.1 | 2026-07-15 | 3 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-19-same-covered-set-prefix-order-equivalence` | 4.8 | 2026-07-19 | 8 | BOUND-COMPLETE |  |
| `2026-07-18-historical-random-versus-geometry-sorted-image2299-screen` | 4.7 | 2026-07-18 | 18 | BOUND-ACTIVE |  |
| `2026-08-27-image2299-output-only-active-competitor-closure` | 4.6 | 2026-08-27 | 2 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-k24-owner-set-score-function-compatibility` | 4.3 | 2026-08-28 | 2 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-25-image2299-geo-sorted-xy-step2444-probes` | 3.6 | 2026-08-25 | 37 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-15-visual-support-counterfactual-commit` | 3.5 | 2026-07-15 | 60 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-21-human13-all-hf-shared-surface-trajectory-credit-vertical` | 3.3 | 2026-08-21 | 189 | BOUND-COMPLETE |  |
| `2026-08-01-sorted-owner-basin-contexts-v2-control-review-final-r6` | 3.3 | 2026-08-01 | 3 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-01-sorted-owner-basin-contexts-v2-control-review-final-r5` | 3.3 | 2026-08-01 | 3 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-01-sorted-owner-basin-contexts-v2-controls-final-r4` | 3.3 | 2026-08-01 | 3 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-01-sorted-owner-basin-contexts-v2-controls-final-r3` | 3.3 | 2026-08-01 | 3 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-22-human13-owner-credit-nk-factorial` | 3.0 | 2026-08-22 | 87 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-01-sorted-owner-basin-contexts-v2-stop-rule6-final-r2` | 2.9 | 2026-08-01 | 3 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-01-sorted-owner-basin-contexts-v2-stop-rule6-final` | 2.9 | 2026-08-01 | 3 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control` | 2.6 | 2026-08-04 | 50 | BOUND-COMPLETE |  |
| `2026-08-27-image2299-owner-set-preserving-gradient-feasibility` | 2.5 | 2026-08-27 | 2 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-complete-row-margin-compilation` | 2.2 | 2026-08-27 | 6 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-25-image2299-natural-and-gt-prefix-free-decode` | 2.0 | 2026-08-25 | 36 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-gt32-two-boundary-compilation` | 2.0 | 2026-08-27 | 7 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-gt32-successor-restoration` | 1.9 | 2026-08-27 | 9 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-02-sorted-owner-basin-task4-control-summary-uncached-merged-v1` | 1.9 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit` | 1.8 | 2026-07-15 | 34 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-27-image2299-native-state-successor-lattice` | 1.7 | 2026-08-27 | 21 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-25-image2299-natural-history-gt-person-prefix-free-decode` | 1.4 | 2026-08-25 | 30 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical` | 1.3 | 2026-08-21 | 197 | BOUND-ACTIVE |  |
| `2026-07-16-human-audited-rare-object-trajectory-genealogy` | 1.3 | 2026-07-16 | 3 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four` | 1.2 | 2026-07-15 | 1 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover` | 1.1 | 2026-07-15 | 2 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-27-image2299-native-guarded-unmatched-row-cleanup` | 1.0 | 2026-08-27 | 1 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-owner-level-near-miss-line-search` | 0.9 | 2026-08-27 | 1 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-25-untouched-terminal-boundary-statistical-analysis` | 0.8 | 2026-07-25 | 4 | BOUND-COMPLETE |  |
| `2026-07-15-selected-transition-batch-precision-prevalence-screen` | 0.8 | 2026-07-15 | 3 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-15-fixed-encoding-downstream-residual-state-portability-gate` | 0.8 | 2026-07-15 | 3 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon` | 0.7 | 2026-07-17 | 2 | BOUND-COMPLETE |  |
| `2026-08-27-image2299-gt32-cross-prefix-retention-projection` | 0.6 | 2026-08-27 | 2 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-19-complete-candidate-row-score-decomposition` | 0.6 | 2026-07-19 | 4 | BOUND-COMPLETE |  |
| `2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response` | 0.6 | 2026-07-15 | 2 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias` | 0.5 | 2026-07-15 | 1 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-25-image2299-xy-same-owner-serialization-sentinel` | 0.5 | 2026-08-25 | 5 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818` | 0.5 | 2026-07-16 | 1 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-21-human13-standalone-owner-credit-probe` | 0.5 | 2026-08-21 | 3 | BOUND-ACTIVE |  |
| `2026-08-01-sorted-owner-basin-task4-control-summary-uncached-timing-ctx7511-26-v1` | 0.4 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid` | 0.4 | 2026-07-15 | 1 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-27-image2299-theta-b-forced-gt19-gt34-union` | 0.4 | 2026-08-27 | 1 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-gt32-cross-prefix-radius-refinement` | 0.4 | 2026-08-27 | 1 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-01-sorted-owner-basin-task4-control-input-plan-final` | 0.3 | 2026-08-01 | 4 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-27-image2299-theta-b-output-only-tied-row-discriminator` | 0.3 | 2026-08-27 | 4 | UNBOUND*(image2299-path-hit) |  |
| `2026-08-27-image2299-theta-b-affine-union-compilation` | 0.3 | 2026-08-27 | 1 | UNBOUND*(image2299-path-hit) |  |
| `2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial` | 0.3 | 2026-07-15 | 1 | UNBOUND | same-name unit dir exists in research-probes (no path citation) |
| `2026-08-01-sorted-owner-basin-input-plan-six-context-smoke-v2` | 0.2 | 2026-08-01 | 4 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-01-sorted-owner-basin-input-plan-six-context-smoke-v1` | 0.2 | 2026-08-01 | 4 | UNBOUND | no same-name unit dir anywhere |
| `2026-07-29-permutation-bundle-k16-val200` | 0.2 | 2026-07-29 | 9 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-04-sorted-prospective-13-image-panel-admission` | 0.1 | 2026-08-04 | 2 | BOUND-ACTIVE |  |
| `2026-07-29-permutation-bundle-k16-val200-max3084` | 0.0 | 2026-07-29 | 8 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_final_20260821T094017Z` | 0.0 | 2026-08-21 | 5 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v5` | 0.0 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v7` | 0.0 | 2026-08-02 | 2 | BOUND-COMPLETE |  |
| `2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v6` | 0.0 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_final_20260821T091729Z` | 0.0 | 2026-08-21 | 5 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_genuine_20260821T093409Z` | 0.0 | 2026-08-21 | 5 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v3` | 0.0 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v4` | 0.0 | 2026-08-02 | 2 | UNBOUND | no same-name unit dir anywhere |
| `2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v2` | 0.0 | 2026-08-02 | 1 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_shape_fix_science_20260821T102033Z` | 0.0 | 2026-08-21 | 7 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_shape_fix_direct_20260821T102202Z` | 0.0 | 2026-08-21 | 7 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_shape_fix_pane_20260821T102430Z` | 0.0 | 2026-08-21 | 7 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_shape_fix_run_20260821T101824Z` | 0.0 | 2026-08-21 | 7 | UNBOUND | no same-name unit dir anywhere |
| `durable-human13_streaming_shape_fix_20260821T101410Z` | 0.0 | 2026-08-21 | 7 | UNBOUND | no same-name unit dir anywhere |
| `logs` | 0.0 | 2026-08-27 | 1 | UNBOUND | no same-name unit dir anywhere |

## Table 3 — `/data/CoordExp/.worktrees/research-probes/outputs/` (123M total)

| root | size | newest mtime | files | class | binding evidence |
|---|---:|---|---:|---|---|
| `outputs/prod/` | 78M | 2026-07-08 | 7 | BOUND-ACTIVE | cited by ~30 current `research/ideas/` and `research/investigations/` unit files (mixed complete/active) plus `openspec/changes/archive/**` accepted-verification evidence |
| `outputs/third_party/` | 42M | 2026-08-03 | 62 | UNBOUND | no hit in `research memories docs/history openspec/changes` |
| `outputs/probes/` | 4.2M | 2026-07-20 | 7 | BOUND-COMPLETE/ACTIVE (mixed) | cited by `research/ideas/qwen3-vl-painted-gt-transcription-probe/...` and `research/investigations/coordexp-experiment-knowledge-handoff/*` (historical-synthesis) plus `openspec/changes/archive/**` |
| `outputs/coordexp_swift/` | 112K | 2026-07-26 | 14 | BOUND-ACTIVE/COMPLETE (mixed) | cited by several current `research/ideas/` and `research/investigations/` unit files plus `openspec/changes/archive/**` accepted infra changes |

## Totals per class (all tables combined, GB)

| class | GB |
|---|---:|
| BOUND-COMPLETE | ~48.5 (34.54 in Table 2 + 14 `pvci_causal_proposal_bridge`) |
| BOUND-ACTIVE | ~19.6 (18.74 in Table 2 + 0.78 `eight-coordinate-bbox-supervision` + 0.08 `outputs/prod`) |
| BOUND-ARCHIVED | ~1.0 (0.14 in Table 2 + 0.877 `pi-lightweight-worker-ablation`) |
| UNBOUND | ~17.1 (15.45 in Table 2 + 1.6 `qwen3-vl-native-text-coordinate-val200` + 0.042 `outputs/third_party`) |
| KEEP regardless of class | `research-probe-infras/` (3.7M, sealed CPU receipts) |

## Largest UNBOUND / BOUND-ARCHIVED candidates the user may name for reclamation

Ranked by size, restricted to roots with **no path citation and no
same-named research unit anywhere** (the highest-confidence UNBOUND set,
3.53 GB / 41 dirs under `qwen3-vl-dense-enumeration`) plus the 3
top-level/other-worktree UNBOUND and BOUND-ARCHIVED roots:

1. `qwen3-vl-dense-enumeration/2026-07-30-iterative-forced-continue-exact-native/` — 1.78 GB, 508 files, newest 2026-07-30, no citation or same-named unit anywhere.
2. `qwen3-vl-native-text-coordinate-val200/` (top-level) — 1.6 GB, zero citations found anywhere.
3. `qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-candidates-six-context-smoke-v2/` — 604 MB, 2 files, no citation or same-named unit.
4. `qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task4-control-candidates-final/` — 521 MB, 2 files, no citation or same-named unit.
5. `pi-lightweight-worker-ablation/` (top-level, BOUND-ARCHIVED) — 877 MB, cited only by an already-`research/archive/`d unit and 3 `memories/notes/`; a legitimate reclaim candidate precisely because its lifecycle is closed and tagged.

Full-precision candidate list (all 41 highest-confidence-UNBOUND dirs plus
the 5 above) is in Table 2's `note` column; the 33 image2299-path-bound and
63 same-name-only-bound dirs are explicitly excluded from this list because
their status is ambiguous or they belong to a still-live direction, not
because they were checked and found safe.

**Nothing was deleted, moved, or modified by this task. All classification
above is advisory; the user names any root to reclaim separately, per
design.md's stated non-goal.**
