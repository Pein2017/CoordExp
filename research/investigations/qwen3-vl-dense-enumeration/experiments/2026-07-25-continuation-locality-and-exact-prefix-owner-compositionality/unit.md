---
title: Continuation-Shift Locality and Exact-Prefix Remaining-Owner Compositionality
description: Bounded no-training study of whether checkpoint continuation changes are specific to trained prefix states and whether verified remaining owners can be recovered and composed after exact model-produced self-prefixes.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality
topic: qwen3-vl-dense-enumeration
status: complete_ready_for_user_discussion
evidence_status: verified_bounded_global_continuation_and_conditional_composition
updated: 2026-07-25
---

# Continuation-Shift Locality and Exact-Prefix Remaining-Owner Compositionality

Executed evidence and the bounded judgment are owned by
[the completed result](results.md).

## Decision

This no-training unit decides whether the completed transition treatment acts
mainly as a local correction at its training states or as a broader
continuation bias, and whether the Source model's remaining-object support can
be recovered and composed after exact model-produced prefixes.

The final task remains one `list all objects` prompt followed by one free
autoregressive completion. Fixed-prefix scoring, forced tokens, teacher-forced
descriptions, and short released rows are diagnostics only. They are not a new
prompt, a production decoding policy, or evidence of final owner-set gain by
themselves.

No optimizer update, new architecture, textual covered-set prompt, one-row-at-
a-time task, or persistent state carrier is authorized in this unit.

## Operational Terms

**Continuation-shift locality** means the degree to which a checkpoint's
change in the log-probability margin between the canonical new-row opener and
`<|im_end|>` is concentrated at exact prefix states used by training rather
than also appearing at comparable untrained row boundaries.

**Exact-prefix remaining-owner compositionality** means two separately
measured abilities at a literal model-produced prefix: first, whether an owner
verified elsewhere in the same image can be generated and geometry-matched
after that prefix under a declared intervention; second, whether appending one
verified new-owner row preserves evidence or released reachability for another
still-uncovered owner. This is an operational diagnostic, not a claim that the
model contains an explicit owner ledger.

## Questions and Strongest Alternatives

### Question One: is the continuation change local?

Compare Source with the existing trained checkpoints at three boundary types:

1. the literal exact prefix used by a row-local training event;
2. a different complete-row boundary from the same image that was not that
   event's trained prefix; and
3. a matched complete-row boundary from an image absent from the event bank.

The primary measurement is the checkpoint difference in canonical row-opener
minus terminal-token log probability. Prefix row count, image object-count
band, natural terminal status, and remaining trusted-owner count are retained
for stratification. Exact-prefix hashes and source artifact identities must be
preserved.

The strongest alternative to a local correction is a global continuation
shift: trained, same-image-near, and untouched boundaries move by similar
amounts after matching or stratification. A smaller same-image-near effect but
a large trained-state effect supports state specificity; it does not by itself
identify which semantic feature defines that state.

### Question Two: can remaining-owner evidence be recovered and composed?

At exact Source-produced row boundaries with at least one verified uncovered
trusted owner, measure a staged intervention ladder:

1. native opener-versus-stop score and native next-row release;
2. force only the canonical new-row opener, then release one complete row;
3. force the opener plus a verified owner's complete description/schema
   prefix, then release coordinates and the row close;
4. score the verified complete owner row as a privileged diagnostic upper
   bound without calling it generated behavior; and
5. when an intervention produces a valid geometry match for a newly covered
   owner, append that literal generated row and repeat the opener, candidate-
   score, and release measurements for another remaining owner.

The primary outcome for one owner is a valid released row matched to that
physical owner. Supporting outcomes are owner-row likelihood by description
and coordinate token group, native and forced stop behavior, invalid rows,
covered-owner repeats, other uncovered owners, and unresolved outputs. The
short-horizon composition outcome is whether adding one verified owner
preserves another owner's relative candidate score or released recovery, not
whether row count merely increases.

The strongest alternatives are that forced continuation produces generic
rows, that description forcing only copies a category while geometry binds a
different instance, or that the first inserted row destroys rather than
preserves remaining-owner reachability.

## Cohorts and Controls

The continuation-locality panel reuses the completed 1,440-image row-local
event bank for trained states. Same-image-near states must be distinct literal
complete-row prefixes and may not reuse the selected event hash. Untouched
controls come from candidate-pool images absent from that bank and are matched
or stratified by prefix depth, object-count band, natural terminal status, and
verified remaining-owner count where support permits. Source and every tested
checkpoint see identical image, prompt, token prefix, runtime, and candidate
tokens.

The compositionality atlas uses Source-produced exact prefixes and trusted
owner geometry from the frozen candidate pool and census. A remaining owner is
positive only when its entity/category and geometry are trusted under the
existing owner-matching contract. Covered-owner controls are retained when a
plausible row is available; other valid uncovered owners and unresolved rows
are neutral, never automatic negatives. Same-category multi-instance examples
receive an explicit stratum because category completion alone cannot identify
the physical owner.

Atlas admission is deliberately conditioned on an owner having at least one
previously observed, verified positive sampled row at that exact prefix. The
atlas therefore measures greedy recovery of sampled-reachable owner evidence;
its rates are not prevalence estimates over every annotated remaining owner.

Source is the primary checkpoint for discovering native support. Transition
step 36 is the primary treatment comparison. The complete-row checkpoints are
included in the continuation-locality measurement because their known output
expansion is directly relevant, but they are not required for every expensive
released compositionality arm.

## Minimal Execution Path

1. Materialize receipt-bound trained, same-image-near, and untouched boundary
   records without retokenizing any persisted model-produced prefix.
2. Materialize a representative four-to-eight-case compositionality smoke,
   including a natural terminal state, a nonterminal state, and a same-category
   multi-instance case when available.
3. Run one real end-to-end Source and transition smoke through model loading,
   exact-prefix replay, scoring, intervention release, parsing, owner matching,
   post-action prefix construction, and immutable receipt writing.
4. Stop if prefix replay, owner identity, row parsing, or post-action
   verification cannot be made exact. Otherwise run the continuation-locality
   panel and scale the compositionality atlas to 300--500 valid
   prefix-by-remaining-owner pairs.
5. Reduce results by boundary type, checkpoint, intervention, prefix depth,
   natural stop, remaining-owner count, and same-category ambiguity. Inspect a
   bounded set of successes and failures before interpretation.

Existing checkpoint loaders, exact-token replay, complete candidate-row
scoring, one-row release, canonical row parsing, and owner matching should be
reused. New implementation is limited to panel materialization, experiment-
local orchestration, and reduction needed for these observations.

## Representative Smoke Gate

The smoke passes only if all selected records satisfy:

- the persisted image, prompt, literal generated prefix, and tokenizer special
  tokens reproduce their declared hashes;
- every prefix ends at a complete row boundary;
- covered and remaining trusted-owner sets recompute from the literal rows;
- each released row is independently parsed and geometry-matched;
- a successful first action is appended using the exact generated tokens, and
  the resulting second boundary is replayed without retokenization; and
- Source and transition runs write immutable receipts with checkpoint,
  precision, generation, and artifact identities.

A valid negative smoke may proceed to scale if the path is exact and the
negative outcome is scientifically interpretable. A broken or ambiguous path
does not.

The representative Source and transition-step-36 smoke passed model loading,
literal prefix replay, boundary and complete-row scoring, native and forced
release, parsing, composite-to-runtime owner identity reconciliation, physical-
owner matching, exact post-action prefix construction, and immutable receipt
writing. Across four selected owner cases per checkpoint, native and forced-
opener release recovered the intended owner in two cases, while complete-
description forcing recovered it in all four. Two selected multi-owner states
also completed both sampled-row and model-realized-row post-action probes. In
the latter, the intended first owner's complete description was forced while
geometry and closure were generated by the model; it is not a fully
free-generated first row.
These observations authorize the declared scale-up but remain smoke evidence,
not prevalence estimates.

## Stop Rules and Claim Boundary

Stop this unit after the two measurements are reduced and discussed. Do not
start training automatically.

Stop expansion early if fewer than 300 valid prefix-by-owner pairs can be
formed without relaxing trusted-owner or exact-prefix requirements, if
post-action prefixes cannot be verified literally, or if owner matching is too
ambiguous to separate category copying from physical-owner recovery. Report
the resulting bounded denominator rather than substituting a looser claim.

The unit may support a local-versus-global continuation judgment, an
intervention-conditional owner-recovery rate, and a one-step remaining-owner
preservation judgment. It cannot establish an explicit covered set, final
greedy owner-set improvement, objective causality from one checkpoint, or the
value of a future training design.

## Scope and Cost

The persistent artifact root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality/`

Small CPU materialization and reduction precede GPU work. The real smoke uses
one or two available GPUs. After the gate passes, independent checkpoint or
panel shards may use the available eight A100 GPUs. The intended scale is a
bounded existing-checkpoint inference and scoring pass, not training.
