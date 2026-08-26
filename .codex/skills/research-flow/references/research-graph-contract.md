# Research Graph and Unit Contract

Use this reference when creating a new investigation, experiment unit, result
record, decision update, or mechanism promotion.

## Semantic Owners

| Surface | Owns | Must not own |
|---|---|---|
| `outputs/research/` | Executed artifacts, receipts, traces, metric primitives | Interpretation or current route choice |
| `research/.../experiments/<unit-id>/` | Evidence-tiered outline or protocol, evidence handles, observed results, bounded verdict | Stable runtime/schema compatibility |
| `research/investigations/` | Competing explanations and synthesis across units | Implementation authorization |
| `research/decisions/` | Current evidence-backed route choice and next discriminator | Mechanism truth or product contract |
| `research/mechanisms/` | Reusable bounded explanations supported across independent units | Single-probe correlations |
| Named infrastructure source, tests, and change | Reusable execution, instrumentation, identity, persistence, recovery, and mechanical acceptance | Cohorts, interventions, estimands, thresholds, scientific outcomes, or claims |
| `openspec/` | Stable reusable implementation and compatibility contracts | Hypotheses, cohorts, thresholds, or scientific verdicts |

## Recommended Investigation Layout

```text
research/investigations/<topic>/
  index.md
  overview.md
  experiments/
    index.md
    <unit-id>/
      unit.md
      results.md   # only after execution closes
      review.md    # only if an audit changes the claim or rerun gate
```

Here `<topic>` is a stable investigation identifier and `<unit-id>` is an
immutable research-unit identifier. The date pattern `YYYY-MM-DD` means a
four-digit year, two-digit month, and two-digit day.

- `index.md`: router, scope, status, reading path, and authority caveat.
- `overview.md`: stable question decomposition, competing hypotheses, evidence
  atlas, and current belief state; never a run log.
- `unit.md`: evidence-tiered executable outline or scientific protocol and the
  unique closure router.
- `results.md`: executed facts and bounded interpretation after evidence closes.
- `review.md`: independent audit findings that narrow, invalidate, or request a
  rerun. Do not create it as an empty ritual.

Use the same shape under `research/ideas/<topic>/experiments/` when the unit
tests a proposed treatment rather than a diagnostic question.

## New Unit Frontmatter

Use this minimum for newly created units; do not bulk-migrate historical units
only to satisfy the new vocabulary.

```yaml
---
title: ...
description: ...
type: investigation          # or idea
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: not_authorized  # not_authorized|authorized_within_goal|authorized_separately
unit_id: YYYY-MM-DD-slug
topic: parent-topic
status: planned              # planned|ready|running|blocked|complete|invalidated|superseded
evidence_status: none        # none|partial|executed_unverified|verified|invalidated
updated: YYYY-MM-DD
---
```

`complete` means execution stopped, evidence scope is fixed, failures are
visible, and interpretation is closed. It does not mean mechanism or
architecture promotion.

Lifecycle meanings:

- `planned`: scientific design or exploratory outline is still being formed;
- `ready`: the evidence-tier-appropriate outline or protocol has enough scope,
  controls, and execution handles to begin;
- `running`: execution has begun under the declared scope;
- `blocked`: execution cannot proceed and the blocker is recorded;
- `complete`: execution and bounded interpretation are closed;
- `invalidated`: contract or evidence failure prevents scientific use;
- `superseded`: a newer named unit replaces this unit's active role;
- `none`: no executed evidence exists;
- `partial`: only part of the declared protocol executed;
- `executed_unverified`: execution exists but receipts or semantics are not yet
  accepted;
- `verified`: evidence passed the declared acceptance checks;
- `not_promoted`: no architecture claim has been promoted from this unit;
- `not_authorized`: no implementation authority has been granted;
- `authorized_within_goal`: an active user-authorized research goal permits
  bounded implementation, controls, and recursive probes within its declared
  objective and cost boundary;
- `authorized_separately`: the user authorized this implementation slice
  outside an active goal.

Initialize `implementation_status` from the active task authority. Do not reset
an already authorized research goal to `not_authorized` merely because a new
unit is created. Architecture promotion and stable-contract work remain
separate decisions.

## Research And Infrastructure Axes

Track a research unit and its supporting infrastructure independently:

| Observation | Research disposition | Infrastructure disposition |
| --- | --- | --- |
| Mechanical path is invalid before an affected contrast is durably recorded | The question is unanswered for that contrast; keep the result neutral and the evidence `none` or `partial` | Preserve the failed run receipt and route the defect to the named infrastructure owner |
| Mechanical path is accepted and the declared contrast yields a null effect | Admit bounded negative evidence if denominator, controls, and uncertainty are satisfied | Record mechanics as accepted without claiming scientific meaning |
| Infrastructure is repaired after a failed run | Keep the old run technically invalid and scientifically unusable for the affected contrast | Validate the repair independently and require a fresh immutable run identifier |
| Conditioning, population, estimand, control, or stop rule changes | Apply the semantic-delta gate and obtain any owning decision; use a new unit when the question changed | Do not disguise the semantic change as an implementation repair |

A technically invalid run is immutable provenance; it does not automatically
invalidate the research unit. Recoverable arm evidence may support only an
unaffected declared contrast. Infrastructure validation can unblock execution
but cannot complete a research unit, promote a mechanism, or rehabilitate
missing evidence.

## Originating Intent And Semantic Delta Gate

Before a unit becomes `ready` or freezes a predicate, cohort, estimand,
control, claim, or stop rule, compare it with the originating user brief or
named criteria. Derived handoffs, reviews, units, and syntheses may route to
those sources but may not silently strengthen them. Record only conditions
that can change scientific meaning:

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Exact behavior or requirement | User brief, evidence, or derived proposal | `scientific invariant`, `loss-conditional`, or `conservative design choice` | Cohort, estimand, control, claim, stop rule, cost, or none | Inherited, approved, proposed, or `needs user decision` |

Classify a condition as a `scientific invariant` directly required by the
originating outcome or safety semantics, `loss-conditional` when required only
by the selected objective or implementation, or a `conservative design choice`
when it strengthens assurance but is not necessary for the broader objective.

A directly cited scientific invariant may be inherited. A loss-conditional or
conservative condition that changes population, estimand, control, claim, stop
rule, material cost, or successor arithmetic keeps the unit `planned` until the
owning decision accepts it. An undefined criterion is a semantic fork, not an
automatic `fix`, unless the originating source determines the answer.

For a census, report nested counts for broad signal, safely usable signal, and
stronger claim-specific subsets. Give separate verdicts for the selected
design and the broader originating objective; the strictest intersection must
not silently become the only feasibility count or successor estimand.

## Terminology and Naming Rule

Every abbreviation, shortened arm name, hypothesis identifier, metric symbol,
coined mechanism name, dataset alias, model alias, and run label must have one
complete declaration at first use or in a local terminology registry. The
declaration must state both the expanded words and the operational meaning.
If the expanded words remain opaque, replace the shorthand with a
behavior-level plain-language name; an abbreviation is optional, not a design
goal.

Examples:

```text
Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)
Local Rescue Rate (`LRR`): probability of detecting an object missed by the
paired full-image single rollout
```

A filename, config key, or historical artifact path is not a declaration.
Avoid ambiguous codes such as an unexplained single letter or opaque two-letter
arm code when a descriptive stable
identifier can be used instead. If a legacy name must remain verbatim, explain
it next to the path.

## Unit Body Contract

Match detail to the evidence tier. An exploratory unit is an executable outline,
not a production pre-registration or a speculative software interface.

### Scientific interpretation gate

Before accepting a positive, negative, or null scientific result:

- report nested source-population, eligible, executed, mechanically valid, and
  analyzable denominators rather than silently using the strictest subset as
  the originating population;
- distinguish natural behavior from oracle, forced-prefix, teacher-forced, or
  retrieval-conditioned behavior, and distinguish precondition or admission
  from realization after the condition is supplied;
- keep technical-invalid, unexecuted, missing-support, unknown, and unmatched
  cases neutral unless the frozen protocol explicitly defines one as the
  scientific outcome; and
- state the transfer needed when an intermediate intervention, proxy, or readout
  differs from the final evaluation surface. Evidence supports the exact
  observed surface until that transfer survives its declared control.

### Exploratory outline

Require only:

1. **Decision and outcome**: the decision at stake, the behavior or capability
   that owns it, and the unit on which that outcome is judged.
2. **Question**: one falsifiable question.
3. **Competing explanation**: the strongest alternative and the control that
   separates it from the working hypothesis.
4. **Primary observation**: the smallest score, trace, visualization, or short
   rollout that changes the next decision.
5. **Alignment**: the proposed intervention or proxy unit, the final evaluation
   surface, the transfer claim between them, behavior that must be preserved,
   ambiguous evidence that stays neutral, and whether the data contains enough
   correctly attributed signal.
6. **Outline**: expected owner surfaces, reused infrastructure, non-goals,
   representative smoke, stop rule, and rough cost. Do not freeze code-level
   interfaces before runtime evidence exists.
7. **Scope**: checkpoint, cases, changed factor, invariants, and decode or
   training semantics needed to interpret the pilot.
8. **Artifact handle**: logical output root and the compact run receipt expected
   from execution.
9. **Terminology**: define local abbreviations and coined names once.

For inference-led case or mechanism studies, prefer a few deliberately selected
representative cases and sample-level review over population metric estimation.
The owning domain skill may define a more specific default range.

Before costly or expanded execution, first make every CPU-discoverable receipt,
serialization, identity, and write-read failure fail before accelerator work.
Then run one minimal real case through every conclusion-bearing mechanical link
from source selection through intervention, durable artifact finalization and
readback, and the evaluator, visualization, or trace that owns the decision. A
helper-level test or in-memory result is not this end-to-end mechanics smoke,
and the smoke proves mechanics rather than model quality.

Only after that smoke passes, run the smallest scientific pilot whose declared
contrast can change the next decision. Expand support, cohorts, replications,
or mechanism matrices only after both gates pass. If a link is bypassed or
represented by a proxy, name it and narrow the claim.

### Decision-grade additions

Add a frozen cohort, primary estimand, minimum meaningful effect, paired
controls, uncertainty plan, safety gate, complete artifact identities, and an
independent evidence audit only after the exploratory observation survives its
control and the semantic-delta gate is closed. Publication- or production-grade
work may then add broader replication, human annotation, stable schemas, and
operational hardening. Do not use randomization journals, exhaustive source
closures, adversarial mutation defenses, or resume machinery to make a
pre-observation exploratory unit appear ready.

After execution begins, record changes that alter scientific meaning,
comparability, scope, or artifact attribution and use a fresh run identifier.
Ordinary implementation repairs do not require a new protocol ceremony. Never
rewrite a declared scope after observing results; label partial execution as
partial evidence.

Record an ordinary implementation repair in the run lineage with its exact
mechanical acceptance. Reuse the unit only when its scientific question and
frozen semantics are unchanged; always use a new run identifier. A repair does
not convert a prior technical failure into scientific evidence.

Before scaling a treatment, distinguish lack of signal supply, optimization
failure, proxy-to-outcome transfer failure, and failure of the intended
mechanism. Do not let the easiest available label, candidate, metric, or
intermediate representation silently redefine the research objective. Use
[Research Alignment Examples](research-alignment-examples.md) for compact
positive and negative patterns.

When closing a non-trivial unit, separate `Observed`, `Supported`, `Ruled out`,
`Unresolved`, and `Not claimed`, then name the next discriminator.

## Independent Advanced-Model Review Gate

An **independent advanced-model reviewer** is an advanced model or agent outside
the primary implementation and internal-audit chain. Pro, Fable, or any future
model may fill this role; the provider name is not part of the research
contract. The reviewer supplies independent scientific criticism and advice.
The research lead remains responsible for the final decision.

Use this gate when at least one of the following is true:

- two or more plausible mechanisms or treatments survive a focused probe and
  internal audit, and the choice changes scientific meaning;
- an unresolved mathematical, statistical, or causal assumption determines an
  expensive training launch, architecture route, scale-up, or claim promotion;
- conclusion-critical evidence is contradictory after one bounded attempt to
  separate the explanations; or
- the user or research lead explicitly requests outside independent judgment.

Do not use it as a substitute for a small local experiment, artifact check,
runtime diagnosis, annotation review, or data census that can answer the
question directly. Do not invoke it as a routine ceremony for every unit, or
with an open-ended request to invent a new architecture before the current
evidence gap is bounded.

Apply the gate as follows:

1. Freeze the decision at stake, the strongest competing explanations, and the
   exact evidence scope. State what remains unknown and what action the review
   may change.
2. Run the cheapest decisive internal observation first. If that observation
   directly answers the question, close the unit without external escalation.
3. Prepare one frozen packet containing a decision brief, an evidence atlas,
   exact source and receipt handles, the requested questions, excluded or
   unavailable evidence, the expected verdict format, and the review stop
   condition. A full chat transcript is not the primary packet.
4. When using multiple reviewers, give them the same core packet independently
   and do not reveal another reviewer's reasoning or verdict. Choose the model
   dynamically for the scientific difficulty; do not hard-code a provider.
5. Require each response to state its assumptions, strongest counterexample,
   recommended disposition, minimal discriminating experiment, and permitted
   claim boundary. Useful dispositions are `proceed`, `narrow`, `probe`,
   `hold`, or `needs user decision`.
6. Reconcile responses against executable evidence. Do not use majority vote as
   proof. Convert material disagreement into a small discriminating experiment
   or an explicit user decision.
7. Treat the review as advisory unless the unit or user explicitly declared it
   a blocking pre-launch gate. A timeout or unavailable reviewer does not
   silently block an otherwise valid unit.
8. Preserve the request and responses as provenance. Create or update the
   concrete unit's `review.md` only when the review changes a claim, rerun gate,
   launch decision, or next discriminator. Update the compass or decision only
   if the route changes, and update project memory only when continuation state
   changes.

## Closeout Consistency Gate

Close a unit from the evidence owner outward:

1. freeze `results.md` or the result section in `unit.md`;
2. update the experiment router and lifecycle status;
3. update the investigation decision or compass only if the route changed;
4. refresh `memories/current.md` only if continuation changed;
5. write a durable handoff only when another session or machine needs one.

These surfaces may summarize the same decision, but they must not become
independent authorities. Keep one current frontier, link to the owning result,
mark superseded routes explicitly, and resolve contradictions before handoff.

A live route means any `current`, `next`, `start here`, fresh-session, or
minimum-reading-path pointer in a compass, active index, decision, or project
memory. Its target must be a tracked owning `unit.md`, `results.md`, research
decision, compass or index, current doc, or stable spec. Handoffs, standalone
agent-review or audit outputs, reviewer packets, transcripts, memory notes,
scratch files, and temporary artifacts may be cited only as provenance; they
must never own the live route. A handoff is consumed transport, not a permanent
route authority.

## Artifact Root

Use:

```text
outputs/research/<investigation>/<unit-id>/<run-id>/
```

Here `<investigation>` is the stable investigation identifier, `<unit-id>` is
the immutable research-unit identifier, and `<run-id>` is one immutable
execution identifier.

Resolve it to an explicit durable absolute root and record that root in the
unit/result. A run identifier is immutable and never reused after partial or failed
execution.

For exploratory evidence, retain only the compact facts needed to attribute and
interpret the observation:

- source commit or dirty-diff identity;
- checkpoint, resolved config, case/input, condition, and seed identities;
- raw output or trace, parser/failure status, and request identifier;
- representative smoke result and primary visual/metric primitive.

Decision-grade evidence adds authored/resolved config hashes, declared changed
and invariant factors, complete terminal failures, raw metric primitives, and
analyzer/scorer identities. Do not duplicate the same identity through nested
seals, receipts, and manifests. Add adversarial runtime attestation only for a
demonstrated threat or ambiguity that could change the scientific conclusion.

A directory path is a handle, not proof. State whether evidence is locally
verified, historical-handle-only, unavailable, metric-bearing, or
mechanics-only.

## Promotion Ladder

1. **Artifact exists**: traceable execution only.
2. **Evidence accepted**: declared scope and evidence-tier-appropriate receipts
   match; observation semantics and material failures are visible.
3. **Research handle promoted**: target effect survives the strongest declared
   control and is reusable by another unit.
4. **Decision updated**: a verified unit changes the current research route.
5. **Mechanism bounded-supported**: at least two independent units or truly
   different intervention families agree, a strong alternative is ruled out,
   and a novel prediction succeeds.
6. **Implementation candidate**: value, native-capability preservation, cost,
   and the smallest reversible implementation are separately justified.
7. **Stable contract**: only durable compatibility-sensitive config, schema,
   runtime, artifact, metric, or interface semantics move to OpenSpec.

Negative evidence may update a decision. A beautiful patch, attention map,
linear probe, single checkpoint, or mechanics smoke cannot promote a mechanism.

## Minimal Verification

- Resolve all local Markdown links.
- Parse new non-router frontmatter.
- Fail review on any unexplained local abbreviation, arm code, hypothesis code,
  metric symbol, or coined name.
- Run `conda run -n ms python scripts/research/check_research_graph.py` for the
  decision layer.
- Check for lifecycle contradictions such as a completed result whose router
  still says `planned`, `ready`, or `running`, or a current decision that names
  a superseded unit.
- Verify referenced artifact receipts/hashes when available.
- Run `git diff --check` on touched files.

Do not expand the graph checker to all historical units until at least two new
investigations have exercised this unit vocabulary without requiring a bulk
migration.
