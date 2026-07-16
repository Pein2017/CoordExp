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

### Exploratory outline

Require only:

1. **Question**: one falsifiable question.
2. **Competing explanation**: the strongest alternative and the control that
   separates it from the working hypothesis.
3. **Primary observation**: the smallest score, trace, visualization, or short
   rollout that changes the next decision.
4. **Outline**: expected owner surfaces, reused infrastructure, non-goals,
   representative smoke, stop rule, and rough cost. Do not freeze code-level
   interfaces before runtime evidence exists.
5. **Scope**: checkpoint, cases, changed factor, invariants, and decode or
   training semantics needed to interpret the pilot.
6. **Artifact handle**: logical output root and the compact run receipt expected
   from execution.
7. **Terminology**: define local abbreviations and coined names once.

For inference-led case or mechanism studies, prefer a few deliberately selected
representative cases and sample-level review over population metric estimation.
The owning domain skill may define a more specific default range.

### Decision-grade additions

Add a frozen cohort, primary estimand, minimum meaningful effect, paired
controls, uncertainty plan, safety gate, complete artifact identities, and an
independent evidence audit only after the exploratory observation survives its
control. Publication- or production-grade work may then add broader
replication, human annotation, stable schemas, and operational hardening.

After execution begins, record changes that alter scientific meaning,
comparability, scope, or artifact attribution and use a fresh run identifier.
Ordinary implementation repairs do not require a new protocol ceremony. Never
rewrite a declared scope after observing results; label partial execution as
partial evidence.

When closing a non-trivial unit, separate `Observed`, `Supported`, `Ruled out`,
`Unresolved`, and `Not claimed`, then name the next discriminator.

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
- Verify referenced artifact receipts/hashes when available.
- Run `git diff --check` on touched files.

Do not expand the graph checker to all historical units until at least two new
investigations have exercised this unit vocabulary without requiring a bulk
migration.
