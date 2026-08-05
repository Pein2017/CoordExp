---
title: Sorted Owner Accessibility Phenotype Census - Independent Review Provenance
description: Provenance record of the independent reviews that hardened the census claim schema and moved the launch gate; not the live route owner.
type: investigation
role: research-review
authority: non_normative_research
unit_id: 2026-08-03-sorted-owner-accessibility-phenotype-census
topic: qwen3-vl-dense-enumeration
status: recorded
updated: 2026-08-03
---

# Independent Review Provenance

**This page is provenance, not protocol.** [`unit.md`](unit.md) owns the live
protocol, and any disagreement between this page and `unit.md` resolves to
`unit.md`. Nothing here authorizes work; it records why the current protocol
looks the way it does.

The census direction was reviewed independently while it was being
implemented. The direction verdict was **PASS with capture HOLD until a
mechanical launch gate**. The deltas below were accepted and folded into
`unit.md`, `capture-rules.json`, and the planner before any capture.

## Accepted P0 deltas

| # | Delta | Where it landed |
| --- | --- | --- |
| P0-1 | The score schema binds `query_suffix_token_ids_sha256` and the scorer re-asserts that the full prefix ends at `BOX_START`, so pre-P0 rows are *mechanically* unjoinable. Old output roots are not mutated; schema version, loader rejection, and a quarantine statement suffice. | `unit.md` canonical-suffix section; plan `score_input_policy` |
| P0-2 | The canonical suffix is **variable length**. Asserting "the last four elements" is wrong for any multi-token description; the assertion is exact tail equality against that category's sealed suffix plus a final `BOX_START`. | `assert_canonical_query_suffix`; multi-token tests |
| P0-3 | A pre-capture `capture-rules.json` artifact with its own digest fixes the suffix, loop rule, minimal-frontier tie rule, alias collapse, admission, and stop policy. Phenotype thresholds stay out of it and are derived on discovery only. | `build_capture_rules` |
| P0-4 | Rank, margin, and posterior keys are strictly `(image, context, category)` over collapsed unique candidates only. | plan `rank_key`; `candidate_collapse` rules |
| P0-5 | Cross-owner identical coordinate tuples **must** collapse once within `(image, normalized_description)` before scoring. Physical identity is `digest(image, category, coord tokens)`; multi-owner provenance is retained; a query group holds exactly one request per unique tuple, and duplicates fail closed. | `build_candidate_bank`; `physical_candidate_id` |
| P0-6 | Strict assignment must be **category-local**. Matching a category-conditioned candidate against every owner in the image let a different-category overlap manufacture spurious ambiguity. The all-category view is retained under the separate name `any_category_assignment_*`. | `strict_assignment` |
| P0-7 | Admission identity is the **exact prefix**, not a suffix-shape class, and the proposal channel is separately admitted. Suffix-shape sharing was rejected because no token-identity-invariance proof exists. | `admission_receipt_id`; capture-rule channels |
| P0-8 | Proposal routing needs a per-`(context, category)` `proposal_route_admission_receipt_id`, because category paths differ in token identity and length; a single context-level observed-prefix receipt does not cover them. The boundary gate stays context-level. | `proposal_route_admission_receipt_id`; three admission channels |
| P0-9 | Hard Qwen3-VL invariants: every model call passes explicit non-`None` `position_ids` (shared mutable `rope_deltas`), no `model.generate()` on the scoring model, no concurrent group threads, fresh `DynamicCache` per group with `cache_length == prefill_length`, and the group backend deleted before the next group. | capture-rule `runtime_invariants` |

## Accepted P1 deltas

| # | Delta | Where it landed |
| --- | --- | --- |
| P1-1 | Owner records carry `tested_context_count` and `non_loop_context_count`; a persistent negative uses only the non-loop primary context family and frontier-tested owners. | `unit.md` frontier section |
| P1-2 | Loop marking is continuous and auditable first: `prior_identical_row_count`, `consecutive_identical_row_run_length`, `repeated_raw_span_sha256`. The `loop_tail` flag is one frozen literal rule and is explicitly not a mechanism label. | `build_context_registry`; capture-rule `loop_rule` |
| P1-3 | `best_over_all_contexts` is diagnostic only; rules consume `best_non_loop_context` and `minimal_frontier_distance_context`. Loop-tail-only support is a separate field. | `unit.md` frontier section |
| P1-4 | The scoring request surface carries distinct observed-prefix and query-prefix digests. | plan query-group rows |
| P1-5 | A per-image target-blind scalar repeat/admission receipt, selected from request-ID digests only. | capture-rule `scalar_reference` |
| P1-6 | The full `1000`-bin `x1` distribution stays in the GPU product despite being analysis-optional, so one-pass capture is followed by CPU-only analysis without a recapture. | capture-rule `diagnostics` |
| P1-7 | The primary owner neighbourhood is the `generator_local_landscape`; strict assignment is a separate owner-identifiable lower bound. `other_owner_strict` candidates are excluded from target support and moved to a collision diagnostic; ambiguous candidates count toward the upper bound only; any `L`/`U` disposition flip is `unresolved`. | `classify_candidate_for_owner`; capture-rule `owner_support` |
| P1-8 | `unmatched_generator_local` enters the **upper bound only, never the lower**: a perturbation designed to move off its owner is still part of that owner's landscape but is not owner-identifiable. | capture-rule `partition_bounds` |

## Accepted support-semantics ruling

The most consequential later ruling replaced **rank-as-support** outright.

| # | Delta | Where it landed |
| --- | --- | --- |
| S-1 | Rank `1` is never evidence of localization support. Rank and margin are a **routing and competition surface only**, published separately and never a support input. | capture-rule `competition_ranks = routing_surface_never_support`; `support_definition.rank_one_as_support = forbidden` |
| S-2 | Support is defined by two continuous local-peak statistics, `peak_lift` and `local_concentration`, computed on `generator_local_max_excluding_other_owner_strict` and evaluated under both ambiguity bounds. | capture-rule `support_definition` |
| S-2a | **Exact formulas, corrected.** `peak_lift = best_logprob - logsumexp(all unique candidate logprobs in the query group) + log(N_unique)` — the best log posterior within the unique `(context, category)` population minus `log(1/N_unique)`, **not** best minus an owner-local reference. `local_concentration = best exclusion-filtered generator-local score - median(that owner's own exclusion-filtered bank scores)`, **not** a normalized-mass share. | `compute_peak_lift`, `compute_local_concentration` |
| S-2b | **Boolean logic, corrected.** Usable support is a *conjunction*: both statistics must reach `threshold + epsilon` under the relevant bound. A persistent negative is therefore "**no** optimistic-`U` context in which **both** clear" — not "each statistic individually below threshold at every context", which an owner clearing different statistics in different contexts would wrongly escape. | `clears_support`, `is_persistent_negative` |
| S-3 | Thresholds are TP-calibrated on **pooled discovery native true-positive owners at the deterministic due boundary**, at fixed quantile `q = 0.10`. | capture-rule `support_calibration` |
| S-4 | `q = 0.05` and `q = 0.25`, and any category-stratified threshold, are **sensitivity/diagnostic only** and never primary, never a disposition. | same |
| S-5 | A stratum with fewer than `20` calibration TPs, or whose `q10` sits within `epsilon` of its minimum, falls back to pooled. A category under `20` is flagged `pooled_underrepresented` and **never** gets a changed threshold. | same |
| S-6 | Fixed `epsilon = 0.002` for single-context support and `0.004` for cross-context deltas. These are constants: observed scalar-reference parity is a **compliance check against** them and never resizes them. | capture-rule `epsilons` |
| S-7 | All twelve raw shards may be front-loaded before any analysis, because blinding is a **hash-DAG** property: capture manifest first, calibration reads discovery digests only, calibration receipt sealed, confirmation binds that exact digest. No retune, no re-derivation on all twelve. | capture-rule `confirmation_blinding` |
| S-8 | `greedy_eligible` is a sealed persistent-negative precondition. The `3` `globally_ambiguous_neutral` owners are retained in every continuous census row but can never close negative and never enter a false-negative denominator: they have no unique identity under the canonical matcher. | capture-rule `non_eligible_owner_policy` |
| S-9 | `native_false_negative` is a sealed persistent-negative precondition. Native true positives are the calibration and positive-control population that *defines* `q10`, so roughly a tenth of them sit below `q10 + epsilon` by construction; labelling them would be circular and would mix the calibration population into the estimated quantity. | capture-rule `native_true_positive_policy` |

| S-10 | Candidate-versus-generating-owner geometry (IoU, centre offsets, extent ratios, areas) must be populated on every admitted logical role and every per-generator provenance entry, measured on the decoded box, preserving multiple generators after cross-owner collapse. A regression during the collapse rewrite had left the helper orphaned and the fields absent. Analysis consumes these; it does not re-derive them. | `candidate_generator_geometry`; `realize_owner_roles`; `generators[].geometry` |

Retention and disposition eligibility are deliberately different things. All
`346` owners keep every continuous row; only `202` (`346 - 141` native TPs
`- 3` ambiguity-neutral) are eligible to close as persistent negatives, and
that same set is the false-negative denominator.

The planner's `build_capture_rules()['owner_support']` is the single frozen
owner of this contract, and every value above is covered by the
`capture_rules_sha256` digest — so resealing the plan after this ruling
changes the plan digest by design.

## Lead dispositions

- **Capture granularity.** An early note implied one process per singleton
  query group (412 model loads). The lead's final choice is **one long-lived
  model/session per image shard with sequential groups**, provided admission is
  re-run explicitly per context/category/channel and never inherited. The
  harness keeps granularity configurable so this can be re-tuned without a
  redesign.
- **Admission sharing.** Accuracy over efficiency for the first capture: exact
  `(context, category, query_prefix_sha256)` admission, no suffix-shape
  sharing.
- **Bank adequacy.** An intermediate threshold keyed on unique assignment
  marked `330` of `346` owners undercovered. The lead rejected freezing it. The
  frozen rule splits **generator-local bank adequacy** (distinct physical
  candidates after collapse, exact anchor mandatory) from **strict-assignment
  coverage** (a separate lower-bound view). `full` at `17`, `adequate_reduced`
  at `12`–`16`, `undercovered_unresolved_only` below `12` or without a
  self-assigned exact anchor. Result: `338` / `8` / `0`. Owners are never
  floored merely for token aliases.
- **Representative smoke image.** The review named `13348`; that is a
  **confirmation** image. The lead corrected the smoke to discovery image
  `6040`, so no held-out evidence is spent before the confirmation rule is
  frozen.
- **`rp1.10`.** Deferred robustness, explicitly **not** a blocker: no canonical
  `rp1.10` native rollout artifact is frozen for this panel, so there is no
  admissible context registry to score it against.
- **Ownership split.** The scorer and the merge/analysis tool are owned by
  dedicated workers; the planner, the visual atlas, and this unit's documents
  are owned here. Schema changes are coordinated by message.

## Advisory caveats carried forward

- The capture gate remains **HOLD** until the mechanical launch gate passes.
  Nothing in this unit claims GPU evidence.
- Cross-owner collapse is implemented and tested, but on this panel it fires
  zero times: no two same-category owners share a coordinate tuple. The rule is
  a correctness guarantee, not an observed reduction.
- The two tokenizer-resolved categories (`bicycle`, `bowl`) rest on a fallback
  validated against all `32` natively observed spans with zero mismatches. That
  validation is evidence about the tokenizer, not about the model.
- Cost is material: `246,067` physical candidate rows across `2,808` query
  groups, with `4134` and `14038` alone contributing about two thirds. Shard
  dispatch is largest-first for this reason.

## Discovery-rule and confirmation review

After capture, an independent `claude-fable-5` xhigh reviewer was restricted to
the discovery half. It reconstructed the calibration thresholds and all `177`
discovery dispositions exactly, verified that no confirmation score artifact
was consumed, and rejected a score-derived competition-margin rule as
quasi-circular with `peak_lift`. It recommended sealing exactly one
score-independent rule:

```text
same_description_owners_ahead_of_frontier >= 8
at the U-bound primary_first_non_loop_minimal_abs_frontier view
for native-FN, greedy-eligible, frontier-tested owners
```

The sealed rule digest is
`45bbe07065670a4291ed7d874fc2a8ca15caee79adf69e3f5fc78ab2e43101f8`.
Confirmation bound it once without retuning.

The same reviewer then independently reconstructed the held-out result. The
phenotype passed its owner/image coverage floor but attenuated from discovery
`RR=2.95` to confirmation `RR=1.45`, with one-sided Fisher `p=0.124` on the
held-out closed dispositions. The review disposition is **directionally
replicated, not confirmed; no promotion**.

The reviewer also reproduced the post-confirmation owner-scale analysis and
recommended closing further observational rule mining on this spent panel.
Apparent scale remains the leading successor hypothesis, but `4134` is a
scale-flat residual mode and two small images reverse the direction. The
accepted next-route recommendation is therefore a fresh, prospectively frozen
single-factor full-canvas resolution intervention, not owner crops and not a
retrofit rule on these artifacts. The main result and all permitted claim
boundaries are owned by [`results.md`](results.md).
