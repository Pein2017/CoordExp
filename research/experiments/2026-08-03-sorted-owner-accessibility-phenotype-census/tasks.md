---
title: Sorted Owner Accessibility Phenotype Census - Task Ladder
description: Ordered implementation and execution tasks with explicit verification handles and honest per-task status for the twelve-image owner accessibility census.
type: investigation
role: research-tasks
authority: non_normative_research
unit_id: 2026-08-03-sorted-owner-accessibility-phenotype-census
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-08-03
---

# Task Ladder

Each task names its verification handle. **A task is marked done only when its
handle has actually been run and passed**, not when its code exists.

Ownership is split across workers. This ladder marks the owner of each task so
no surface is edited twice.

| Task | Owner | Status |
| --- | --- | --- |
| A. Unit skeleton and protocol | this worker | done |
| B. CPU planner, incl. the frozen support-calibration contract | this worker | done, tests green |
| C. Scorer / executor | dedicated scorer worker | done; real-HF smoke and batch ladder passed |
| D. Merge and analysis | dedicated merge worker | done; discovery/confirmation hash-DAG verified |
| E. Visual atlas, reading the merge owner-summary schema | this worker | done; real artifacts rendered |
| F. Focused test suite | all owners | done; `353 passed`, Ruff clean on the fixed tree |
| G. Plan seal | this worker | done; immutable run `20260803T065743Z` |
| H. Representative smoke on `6040` | scorer worker | done; admission and merge smoke passed |
| I. Twelve-image capture | scorer worker | done; `12/12` complete, no quarantine |
| J. Merge, analysis, atlas render | merge + this worker | done; discovery, confirmation, presentation and 61-figure atlas complete |
| K. Interpretation | lead | done; bounded verdict in [`results.md`](results.md) |

## Closure

The task ladder is complete. The authoritative executed artifacts live under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z/`.
The confirmation phase bound calibration `9dd6d764...` and discovery rule
`45bbe070...` exactly once without retuning. The unit remains observational;
completion does not authorize a treatment or architecture.

## A. Unit skeleton and protocol — done

`unit.md` states all fifteen contract items, the verified `346`/`343` census
shape, the `6`/`6` split, the variable-length canonical suffix as literal token
order, the admission channels, the runtime invariants, and the stop rules.
[`review.md`](review.md) records the accepted review deltas as provenance.

*Handle:* the protocol document exists and is internally consistent with the
planner's sealed `capture-rules.json`.

## B. CPU planner — done, tests green

`scripts/research/build_sorted_owner_accessibility_census_plan.py` binds and
verifies every immutable input digest, and builds the image, owner, category,
context, candidate, query-group, native-sidecar, and shard registries plus a
sealed `capture-rules.json` and a self-reconstructing `receipt.json`.

It also freezes the support-calibration contract in
`build_capture_rules()['owner_support']`, which is the single owner of:
`peak_lift`/`local_concentration` definitions; pooled discovery native-TP
due-boundary calibration; `primary_quantile = 0.10`;
`sensitivity_quantiles = [0.05, 0.25]` diagnostic-only;
`support_epsilon = 0.002`; `cross_context_delta_epsilon = 0.004`;
`category_contribution_min = 20` with the `pooled_underrepresented` flag; the
confirmation hash-DAG blinding requirements;
`unmatched_generator_local = upper_bound_only`; and
`competition_ranks = routing_surface_never_support`.

*Handle (run):* `--dry-run` over all twelve images reports `346` owners, `343`
greedy-eligible, `141` native true positives, the `6`/`6` split, `412` contexts
(`82` loop-tail), `2843` query groups (`2808` admitted), `5858` physical
candidates, and per-shard `estimated_work_units` in largest-first order.
Tests assert every frozen support value exactly and that tampering with one
moves `capture_rules_sha256`.

**Note:** freezing this contract changes the sealed plan digest by design; the
plan must be resealed (task G) before capture.

## C. Scorer / executor — ceded

Owned by the dedicated scorer worker at
`scripts/research/score_sorted_owner_accessibility_census_shard.py` and its
matching test. This worker created an initial version of that file immediately
before the cede instruction arrived and has not touched it since; it is
preserved in place for that worker to inspect or replace.

## D. Merge and analysis — ceded

Owned by the dedicated merge worker at
`scripts/research/merge_sorted_owner_accessibility_census_shards.py` and its
matching test. It must accept the full singleton-context shard set, fail any
score row lacking a covering `admission_receipt_id`, declare the global stop
above two quarantined images, and bind the frozen discovery-rule digest without
retuning.

## E. Visual atlas — done, tests green

`scripts/research/visualize_sorted_owner_accessibility_visual_atlas.py`
produces the owner map, proposal map, localization landscape/candidate map,
owner cards, and feature overview plus `visual-manifest.json`, reading only
JSONL artifacts and receipts.

Owner cards and the feature overview read the merge tool's
`owner-summaries.jsonl` and present `peak_lift`, `local_concentration`, and the
calibrated support status verbatim, with routing rank and margin kept in a
separate block. The atlas **never** applies a quantile, compares a statistic to
a threshold, or reads rank `1` as support.

*Handle (run):* focused tests assert JSONL-only provenance, manifest
completeness over all five products, within-context normalized confidence,
short-ID labels with a side legend, fixed-padding crops that cover the
competition neighbourhood, refusal of quarantined-shard evidence, that the
`x1` distribution is never rendered, and that support/thresholds are copied
rather than recomputed.

## F. Focused test suite — done, green

`tests/research/test_build_sorted_owner_accessibility_census_plan.py` (79) and
`tests/research/test_visualize_sorted_owner_accessibility_visual_atlas.py`
(35).

*Handle (run):* `conda run -n ms python -m pytest` over both files: **114
passed**; `ruff check` over all four owned files: clean.

The visual atlas is aligned to the merge schema as it stands while merge
implementation is still in progress; if that schema moves, the reader and its
fixtures follow it.

## G. Plan seal — complete

The planner resolved `bicycle` and `bowl` with the pinned tokenizer and sealed
`plan/` under immutable run ID `20260803T065743Z`.

*Handle:* `receipt.json` reconstructs its own digest, `capture-rules.json`
reconstructs its own digest, and no category remains `blocked`.

## H. Representative smoke on image `6040` — complete

The smallest **discovery** shard (`15` owners, `4` categories, `11` contexts,
`2,772` candidate rows) passed. It was deliberately not a confirmation image.

*Handle:* a shard receipt with per-context/channel admission receipts and
recorded scalar-versus-cached bounds.

## I. Twelve-image capture — complete

Shards were dispatched largest-first with one long-lived session per image and
sequential groups. All twelve completed; none was quarantined, missing, or
incomplete.

*Handle:* twelve shard receipts, at most two quarantines.

## J. Merge, analysis, and visual atlas render — complete

*Handle:* split-separated merge receipts, `confirmation-report.json`, and the
combined `visual/combined/visual-manifest.json` covering `61` figures.

## K. Interpretation — complete

[`results.md`](results.md) is written against the captured artifacts and is
bound by contract item 15. It closes the census without promoting a causal
mechanism, training objective, or architecture.

**Replay note.** Producer scripts deleted from `research-probes` on 2026-08-28 (reclaim-research-probes-lifecycle); replay them from tag `research-base-v2`: `git worktree add <tmp> research-base-v2`.
