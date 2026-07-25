---
title: Existing-Checkpoint Transition Mechanism Decomposition and Robust Evaluation
description: Exploratory no-new-training unit that separates robust free-rollout evidence, continue-versus-stop movement, conditional owner selection, row realization, full-row score normalization, and held-out owner churn for Source and transition step 36.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-25-existing-checkpoint-transition-mechanism-decomposition
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: none
updated: 2026-07-25
---

# Existing-Checkpoint Transition Mechanism Decomposition and Robust Evaluation

## Decision and Outcome

This unit decides what the completed first-divergence transition step-36
checkpoint changed before any replication training, objective redesign, or
architecture work is launched.

The decision-owning outcome remains the original task: one `list all objects`
prompt, one free autoregressive completion, and final unique trusted physical-
owner coverage with useful Source owners, row validity, geometry, low
duplication, and natural stopping preserved. A fixed-prefix likelihood, forced
continuation, or teacher-forced row is a diagnostic proxy and is never called
final-set improvement by itself.

The unit ends with one of four dispositions:

1. `gate-dominant`: the treatment mainly changes continue versus stop;
2. `post-continue-improvement`: an advantage remains after continuation is
   fixed, through owner choice or row realization;
3. `normalization-or-dose-dominant`: the apparent objective difference is
   substantially explained by summed-versus-mean scoring or effective token
   dose;
4. `unresolved`: the bounded observations cannot separate these explanations.

These are unit-local descriptions of observed behavior, not names of model
modules or architecture components.

## Originating Intent and Semantic Delta

| Condition | Source | Class | Decision effect | Disposition |
|---|---|---|---|---|
| Preserve the original one-prompt, one-completion `list all objects` outcome | User direction and investigation compass | scientific invariant | final evaluation and claim | inherited |
| Report gains, retained owners, losses, prediction count, duplication, geometry, and stopping rather than owner net alone | User-approved Phase Zero scope | scientific invariant | evaluation | approved |
| Treat fixed-prefix and forced-continuation measurements as diagnostics | research alignment contract | scientific invariant | claim boundary | inherited |
| Compare Source and transition step 36 at identical prefixes and runtime surfaces | current mechanism question | loss-conditional | causal interpretation | approved |
| Use the current Intersection over Union 0.50 owner match as primary and additional thresholds only as sensitivity analyses | existing transfer comparison | conservative design choice | robustness interpretation | inherited primary; sensitivity does not replace it |
| Begin with one real case and a small case panel before any broad forced-continuation fan-out | exploratory evidence tier | conservative design choice | cost and stop rule | approved within the active goal |
| Do not start new training in this unit | user-approved Phase Zero boundary and active goal | scientific invariant | execution scope | inherited |

No strict census predicate, natural-order alias requirement, one-row-at-a-time
prompt, textual covered-set prompt, state carrier, or final architecture is
introduced here.

## Questions and Strongest Alternatives

### Question One: robust free-rollout effect

Does the Source-to-transition owner-set direction remain visible after paired
per-image reporting, explicit length-stop sensitivity, and removal of decode
tail domination?

The strongest alternative is that a few very long or truncated rows create the
aggregate signature while typical images do not improve.

### Question Two: where the local effect enters

At the same exact prefix, does transition step 36 change:

1. the score of the canonical new-row opener relative to `<|im_end|>`;
2. which physical owner is preferred after continuation is fixed; or
3. the probability, validity, and geometry of completing the selected row?

The strongest alternative to conditional uncovered-owner selection is a
generic continuation or row-writing improvement that raises covered,
uncovered, and unsupported legal rows together.

### Question Three: full-row normalization and fixed budget

Do the complete-action pairwise arms remain qualitatively worse when their
candidate scores are reported as sequence sums, per-token means, and separate
description/schema and coordinate group means, and when free outputs are
compared at common row or token budgets?

The strongest alternative is that sequence length and effective token dose,
rather than supervision span itself, account for much of the difference from
the transition checkpoint.

### Question Four: held-out owner churn

Among the 46 geometry-matched gains and 39 losses on heldout-128, how many are
true physical-owner changes versus category alias, localization threshold,
same-owner relabeling, prediction-order effects, or unresolved annotation
cases?

The strongest alternative is that the `+7` owner net is mostly matching churn
rather than changed physical-object coverage.

## Existing Evidence and Checkpoints

The owning predecessor is [Prefix-Local and On-Policy Final Owner-Set
Training](../2026-07-24-prefix-local-and-on-policy-owner-set-training/results.md).

The Source checkpoint is the production step-4,887 description-first,
geometry-sorted pure-cross-entropy adapter. The treatment is the
first-divergence transition checkpoint after 36 optimizer updates. Its direct
branch term compares the canonical row opener against `<|im_end|>` at one exact
prefix, and its positive-row continuation term averages schema/description and
coordinate token losses by group. The branch term does not encode a physical
owner identity; the selected positive row and its continuation tokens do.

The existing matched transfer evidence uses Hugging Face inference, batch size
4, 3,084 generated tokens, greedy decoding, temperature 0, top-p 1, repetition
penalty 1, the original prompt, and one completion. Development has 256 images;
heldout has 128 images. Heldout is the primary outcome cohort for this unit and
development is supporting evidence. Heldout is now an analyzed exploratory
cohort, not a future blind confirmation set.

## Phase Zero Execution Outline

`Phase Zero` means existing-checkpoint analysis and short inference probes with
no parameter updates.

### Lane One: robust evaluation from persisted artifacts

For development-256 and heldout-128, report:

- all-image aggregate owner, prediction, geometry, invalid-row, and stop
  measures;
- per-image median and a declared symmetric trimmed mean;
- the paired subset where both arms stop naturally, while retaining the full
  cohort as the primary free-rollout estimate;
- gained, retained, lost, and missed-by-both trusted owners;
- unique matched-owner yield per valid prediction;
- strict physical-owner duplicate-candidate count and rate per valid
  prediction;
- the largest per-image influences and fixed generated-token-budget curves;
- the leave-one-image-out net range and concentration by object category;
- owner matching at Intersection over Union 0.50 as primary, with 0.30 and 0.75
  reported only as sensitivity analyses.

The primary robust observation is whether owner direction and prediction
efficiency agree between the full cohort, paired natural-stop subset, and
trimmed per-image descriptions. These analyses do not silently delete length
stops from the main result.

### Lane Two: same-prefix mechanism decomposition

The first real smoke reuses one historical human-refined state with an already
verified stop intervention and passes model loading, exact-prefix replay,
token scoring, released generation, parsing, owner matching, and artifact
writing end to end. The exploratory current-checkpoint panel then uses four to
eight deliberately selected exact prefixes before any wider prevalence run.

Source-produced exact prefixes are the primary fixed-state estimand because
they ask what each checkpoint does at the same Source-visited state. If this
panel shows that treatment-visited state is decision-relevant, a secondary
cross-replay scores both checkpoints on both Source-produced and treatment-
produced prefix banks. A Source-prefix result alone is not called an on-policy
treatment mechanism.

At every prefix, keep the image, prompt, prefix token identifiers, tokenizer,
special-token embeddings, runtime precision, and generation policy fixed while
changing only the checkpoint or declared intervention.

1. **Continue-versus-stop measurement.** Record the full-precision log-
   probability difference between the canonical row opener and `<|im_end|>`.
2. **Conditional owner selection.** Teacher-force exactly the canonical
   `<|object_ref_start|>` token as the primary operational meaning of
   “continuation is fixed,” then score at
   least one verified uncovered-owner row, one plausible covered-owner row,
   and another verified uncovered-owner row when available. Keep unresolved
   or unsupported candidates neutral rather than naming them negative.
3. **Row realization.** For the same intended owner, report description/schema
   and coordinate token likelihoods separately, then release generation after
   the smallest owner-identifying prefix supported by the case. Measure row
   validity, matched physical owner, and geometry.
4. **One-time forced continuation.** Where the native top choice is stop, the
   primary arm forces the canonical row opener and then lets the model generate
   one row. A separate historical-comparability control masks only that first
   terminal choice and lets the model choose its best nonterminal token. When
   that token is the canonical row opener, the two interventions are recorded
   as behaviorally equivalent for that prefix. Classify the released row as a
   new uncovered owner,
   covered-owner repeat, same-category localization failure, unsupported or
   unresolved candidate, invalid geometry, malformed row, or no completed row.

Greedy release answers which branch is top-ranked. A small fixed set of
low-temperature sampling seeds may estimate conditional outcome frequencies
only after the deterministic smoke is valid; it is not a population prevalence
claim.

### Lane Three: complete-action score and budget reanalysis

Use exact candidate paths and persisted token traces where available. For each
candidate or completed generated row, report without changing the underlying
token evidence:

- summed autoregressive log probability;
- mean log probability per target token;
- mean schema/description-token log probability;
- mean coordinate-token log probability;
- token count and row phase boundaries.

The two group means receive equal diagnostic weight, matching the transition
continuation reducer. All underlying token counts and the unweighted token
mean remain visible. This diagnostic does not redefine the complete-action
objective that was actually trained.

Terminal evidence remains separate from complete-row normalization. Do not
construct a candidate-set probability distribution from independently scored
rows.

For free rollouts, rebuild owner comparisons at common complete-row counts and
fixed generated-token cutoffs using only rows whose closing box token is
present by the cutoff. The all-token-budget free rollout remains the primary
behavioral result.

Where the pairwise and owner-conditioned event banks share an identical
singleton candidate action, use that common projection for the direct
normalization comparison. Additional aliases in the owner-conditioned bank are
reported separately rather than changing the common denominator. If retained
token likelihoods are insufficient, run only a bounded exact-event rescoring
pass; do not infer missing likelihoods from rollout counts.

### Lane Four: held-out gained/lost physical-owner review

Materialize a review ledger for all 85 geometry-derived changed-owner
references. Each entry must show the original image, trusted owner category and
box, Source predictions, transition predictions, attributed matches and
overlaps, enlarged crops, and raw output order.

Initial human disposition is arm-blinded where the packet permits it and is
recorded on two axes:

- entity/category: real gain or loss, duplicate, category alias or disagreement,
  unsupported candidate, or uncertain;
- geometry: acceptable, localization error, neighboring-instance or mixed-
  extent error, or uncertain.

Officially unmatched predictions remain unresolved until reviewed; they are
not automatically hallucinations. The review reports both the original
geometry-derived ledger and the human-refined ledger.

## Alignment and Claim Boundary

The same-prefix measurements intervene on one local action or row. They can
identify where checkpoint behavior differs, but they do not establish that the
model carries an explicit covered set. A treatment advantage after forced
continuation supports a post-continue conditional effect only for the tested
prefixes. It becomes evidence for usable enumeration only when the original
free one-completion owner set improves without owner exchange, duplicate or
invalid-row bursts, geometry regression, or unsafe stopping.

The final result must preserve these distinctions:

- existence versus prevalence;
- log-probability movement versus released behavior;
- entity selection versus coordinate realization;
- geometry-derived owner change versus human-confirmed physical-owner change;
- checkpoint-level evidence versus objective-recipe evidence.

The unit does not require zero lost owners, every image to improve, or a
confidence interval to exclude zero. Those would strengthen the user-owned
success semantics. Uncertainty and owner exchange instead bound the claim and
the next decision.

## Reused Infrastructure and Expected New Surfaces

Reuse the current physical-owner matcher and strict duplicate-candidate logic,
the historical row-boundary terminal suppressor, fixed-prefix forced-row
generation, complete-candidate-row token scoring, current Hugging Face session,
and existing detection comparison renderer.

Add only experiment-local orchestration and reducers needed to:

- join raw and scored rows by exact row identifier;
- compute the robust paired summaries and fixed-budget curves;
- emit the fixed-prefix checkpoint-by-intervention receipt;
- materialize the 85-owner review ledger and visual packet.

Do not promote a new shared inference interface during the first smoke.

## Representative Smoke and Scaling Gate

The first smoke must cover one exact prefix through:

1. Source and transition checkpoint loading;
2. native continue/stop scoring;
3. one forced-continuation condition;
4. at least two complete candidate rows with unequal token lengths;
5. separate summed, token-mean, description/schema, and coordinate scores;
6. released row parsing and physical-owner matching;
7. one compact receipt and a review image when matching changes.

Scale to the small panel only if exact prefix hashes, checkpoint identities,
token grouping, stop handling, parser behavior, and owner attribution all pass.
Scale beyond sixteen cases only if the small panel exposes a stable difference
whose prevalence would change the next training decision.

## Stop Rules

Stop and revise the probe if checkpoint composition, prompt, special-token
payload, prefix token identifiers, image processing, precision, or batch policy
cannot be held comparable.

Stop a lane when its cheapest observation resolves its declared alternative:

- if robust reanalysis reverses the owner direction, do not use the current
  aggregate result to motivate mechanism promotion;
- if forced continuation removes the Source-to-transition difference, classify
  the bounded effect as gate-dominant and do not claim owner selection;
- if uncovered, covered, and unsupported rows move together, classify the
  effect as generic row continuation or realization rather than uncovered-
  owner selection;
- if normalized and fixed-budget reanalysis removes the complete-action
  contrast, route the next training control to normalization and dose rather
  than supervision span;
- if the human-refined held-out owner net becomes zero or negative, preserve
  the directional checkpoint observation but do not call it physical-owner
  improvement.

Complete this unit after all four lanes have either verified evidence or an
explicit bounded stop. Then write `results.md`, update the experiment router
and compass if the route changed, refresh project memory, and stop for user
discussion before any new training.

## Artifact Roots and Rough Cost

Logical and resolved durable root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-25-existing-checkpoint-transition-mechanism-decomposition/
```

Use immutable run identifiers under that root for robust reanalysis,
fixed-prefix smoke and panel, complete-action score reanalysis, and held-out
owner review. Existing-artifact reducers are CPU-scale. The one-case and small
fixed-prefix panels require short Source and transition checkpoint inference,
not training. A wider sampled panel requires a separate scale decision inside
this unit after the small-panel gate.

## Non-Goals

- no new optimizer updates or long training;
- no full-pool confirmation, seed replication, or best-checkpoint selection;
- no one-row-at-a-time or textual covered-set prompt;
- no state carrier, object slot, external detector, or final architecture;
- no claim that forced continuation is a production decoding policy;
- no relabeling of unmatched predictions as hallucinations without review.
