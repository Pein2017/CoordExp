---
doc_id: progress.diagnostics.stage1-coord-basin-duplication-mechanism-2026-04-11
layer: progress
doc_type: diagnostic
status: active
domain: research-history
summary: Inference-only synthesis of the current best-supported Stage-1 duplication mechanism: a coordinate basin and weak x1/y1 local escape barrier, with late history overwrite acting as an amplifier.
tags: [progress, diagnostics, stage1, duplication, rollout, coord, mechanism]
updated: 2026-04-11
---

# Stage-1 Coordinate-Basin Duplication Mechanism

## Why this note exists

The current duplication-collapse investigation has moved beyond the earlier
surface explanation that "the model attends to recent text instead of vision."
The best-supported mechanism is now narrower and more actionable:

- the primary separator is early coordinate escape at `x1` / `y1`
- the likely root issue is a local coordinate basin or weak escape barrier
- late history overwrite remains important, but current evidence supports
  treating it as a secondary amplifier rather than the sole cause

This note records the current convergence point so the conclusion is not lost
inside per-run artifact bundles.

## Current best-supported diagnosis

The working mechanism is:

1. Stage-1 soft/distributional coordinate supervision appears to preserve a
   smooth local spatial manifold around nearby same-desc solutions.
2. Under teacher forcing, that local smoothness can look acceptable because
   expectation-style geometry and soft-target objectives do not require the
   model to strongly reject nearby alternatives.
3. During rollout, the next object often begins from a history-biased state.
   That part alone is not yet pathological.
4. Duplication becomes likely when the first coord decisions, especially
   `coord_x1` and `coord_y1`, fail to evacuate probability mass away from the
   previous or nearby local bbox neighborhood.
5. Once that early escape fails, recent generated history and prior coord spans
   make the chosen basin self-reinforcing.

Short version:

- the root cause is best framed as a **coordinate basin / local escape barrier**
- late history overwrite is a **basin stabilizer**
- same-desc duplication is one visible expression of that basin

## What changed relative to the earlier hypothesis

The earlier investigation established that duplicated cases often place more
late-layer attention on generated history than on vision tokens. That finding
still matters, but it is no longer the main explanation.

Healthy same-desc transitions can also show noticeable late history overwrite.
What separates them from failures is that healthy cases still make the coord
posterior sharper and move out of the previous/local neighborhood quickly.

That is why the primary diagnosis surface is now:

- `predicted_object` vs `exact_duplicate`
- measured at early coord phases
- with `x1` / `y1` neighborhood escape taking priority over aggregate attention
  summaries

## Main evidence so far

### 1. Focused contrastive probes support x1/y1 escape as the main separator

The strongest currently validated contrastive runs are:

- Soft 4B focus:
  - [`report/summary.json`](../../research/duplication_collapse_focus/duplication-collapse-contrastive-focus-softce4b/report/summary.json)
  - [`probe/case_rows.jsonl`](../../research/duplication_collapse_focus/duplication-collapse-contrastive-focus-softce4b/probe/case_rows.jsonl)
  - [`compare/case_rows.jsonl`](../../research/duplication_collapse_focus/duplication-collapse-contrastive-focus-softce4b/compare/case_rows.jsonl)
- Center focus:
  - [`report/summary.json`](../../research/duplication_collapse_focus/duplication-collapse-contrastive-focus-center/report/summary.json)
  - [`probe/case_rows.jsonl`](../../research/duplication_collapse_focus/duplication-collapse-contrastive-focus-center/probe/case_rows.jsonl)
  - [`compare/case_rows.jsonl`](../../research/duplication_collapse_focus/duplication-collapse-contrastive-focus-center/compare/case_rows.jsonl)

Across those runs:

- duplicated cases tend to keep `coord_x1` / `coord_y1` broad, high-entropy, or
  sticky to the previous/local neighborhood
- healthy same-desc controls can still show history-overwrite patterns, but
  they build a sharper coord posterior and escape the local basin

Interpretation:

- late history-overwrite is not sufficient on its own
- early coord escape is the more specific and robust discriminator

### 2. Predicted-object vs exact-duplicate is the strongest causal comparator

The broadest current cohort comparison is driven by:

- [`compare_control_vs_failure.py`](../../temp/compare_control_vs_failure.py)
- [`center_expanded_control_vs_failure.json`](../../research/duplication_collapse_control_compare/center_expanded_control_vs_failure.json)
- [`ceproxy_expanded_control_vs_failure.json`](../../research/duplication_collapse_control_compare/ceproxy_expanded_control_vs_failure.json)

These comparisons improved on the earlier `gt_next`-based framing.

Why this matters:

- `gt_next` is useful supporting context, but it is not the object the model
  actually chose under rollout
- `predicted_object` vs `exact_duplicate` directly tests whether the chosen
  continuation escaped the duplicated local basin

Current cross-case pattern:

- healthy controls usually reduce `x1` / `y1` previous-neighborhood mass
  relative to the exact duplicate candidate
- failure cases often do not reduce that mass consistently, especially on
  `y1`

This is the strongest cohort-level evidence for the current mechanism.

### 3. The bad state is broader than literal bbox copying

The current artifacts support a broader conclusion than "the model copied the
last box."

Observed failure expressions include:

- same-desc near-copy loops
- cluster-drift loops that stay inside one local object family without exact
  previous-box copying
- semantic escape cases that avoid visible duplication only by drifting into a
  different failure mode

Interpretation:

- the underlying issue is a local spatial basin
- exact duplication is one manifestation of that basin, not the full story

## What remains uncertain

### Pure CE

We should stay conservative here.

The current on-disk CE-side checkpoints are continuation-style proxies, not yet
clean token-compatible from-scratch pure-CE references for the active pipeline.
That means the current evidence supports:

- "soft/distributional coord supervision appears to lower the local escape
  barrier"

more strongly than it supports:

- "pure CE has already been proven to solve duplication"

When a compatible pure-CE checkpoint becomes available, it should be evaluated
with the same `predicted_object` vs `exact_duplicate` and `x1` / `y1` escape
probes rather than with a new protocol.

### Attention as a primary cause

Late history overwrite is real and should continue to be logged, but current
evidence does not justify treating it as the primary cause without stronger
controlled intervention results that actually change decoded behavior.

## Practical implications

### For diagnosis

Prefer these signals first:

- `coord_x1` / `coord_y1` previous-neighborhood mass
- `coord_x1` / `coord_y1` entropy and top-1 probability
- `predicted_object` vs `exact_duplicate` score margin

Treat these as supporting signals:

- final-layer history-vs-vision attention imbalance
- prior-coord-token attention concentration
- whole-object average entropy

### For future objective changes

The most plausible objective-side target is not generic "more vision attention."
It is stronger **same-prefix instance discrimination** in coord space so that
the true next instance cleanly outranks local copied alternatives at the first
coord steps.

## Current recommendation

Until a compatible pure-CE checkpoint arrives, the safest wording is:

- Stage-1 soft/distributional coordinate supervision appears to create or
  preserve a smoother local coord basin
- rollout fails when early `x1` / `y1` decisions do not escape that basin
- late history-overwrite then makes the basin sticky

That is the current converged mechanism-level diagnosis.
