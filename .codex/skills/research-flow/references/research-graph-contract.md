# Research Graph and Unit Contract

Use this reference when creating a new investigation, experiment unit, result
record, decision update, or mechanism promotion.

## Semantic Owners

| Surface | Owns | Must not own |
|---|---|---|
| `outputs/research/` | Executed artifacts, receipts, traces, metric primitives | Interpretation or current route choice |
| `research/experiments/<unit-id>/unit.md` or exact preserved protocol | Evidence-tiered outline/protocol, frozen question and contrast | Live lifecycle updates or a growing execution notebook |
| `research/experiments/<unit-id>/state.json` | Current lifecycle, evidence/disposition axes, latest user boundary and result/protocol pointers | Metrics ledger or permission inferred from old grants |
| Accepted result and its immutable receipt | Observed outcomes, denominators, bounded verdict and evidence handles | Stable runtime/schema compatibility |
| `research/questions/` and `research/story.md` | Competing explanations, evidence-linked beliefs, bounded mechanisms, route choices and research transitions | Implementation authorization or copied volatile status tables |
| `research/alternatives.md` | Important unanswered ideas with predecessors and reopening conditions | A parallel result atlas or standing execution queue |
| Named infrastructure source, tests, and change | Reusable execution, instrumentation, identity, persistence, recovery, and mechanical acceptance | Cohorts, interventions, estimands, thresholds, scientific outcomes, or claims |
| `openspec/` | Stable reusable implementation and compatibility contracts | Hypotheses, cohorts, thresholds, or scientific verdicts |

## Flat Research Root and Reading Path

The registered research base's `research/CONVENTIONS.md` owns the layout and
maintenance contract. Resolve that Project first; the Skill's installation
checkout is not automatically the research base. Begin at `research/index.md`.

The whole repository serves one research topic. Use the flat research root:
`index.md` for frontier plus routing, `story.md`, `glossary.md`,
`alternatives.md`, `questions/`, and `experiments/catalog.jsonl` plus live units.
Old OKF idea/decision/mechanism/archive buckets and the topic wrapper are retired.
Valuable scientific distinctions enter question pages; original sources remain
under `docs/history/`. No permanent compatibility alias or empty future category.

The fast reading path is current context plus its state/result; the deeper path
adds the story, relevant question pages and decisive original records. Preserve
ideas, counterexamples and exact source handles rather than forcing every file
to repeat all background. Before proposing a unit, identify its closest tested
predecessor, remaining uncertainty, changed factor and reopening condition.

`unit.md` is the proportionate design/protocol, frozen when execution begins.
`state.json` is the current lifecycle owner. Accepted results own facts, while
questions own interpretation. A review exists only for an actual claim/rerun
risk. Handoffs are integrated transport and then archived, never live frontiers.
Raw source provenance goes to `docs/history/`; maintained code and executed
artifacts retain their separate owners. Preserve exact path/hash dependencies
or explicitly delimit archived-source versus executable-replay compatibility.

## New Unit Identity and Current State

Use light stable protocol metadata: title, unit_id, question identity,
role, evidence/authorization source and freeze identity when applicable. The
protocol body owns the actual scientific contract. Do not embed independently
maintained current lifecycle fields in a frozen launch snapshot.

Use the state schema defined once in `research/CONVENTIONS.md`: lifecycle,
evidence and scientific disposition are separate axes, with exact protocol,
result and state-source paths, an as-of point, latest user boundary and next
action. A paused task may have accepted evidence and an incomplete stage.
Closure never implies mechanism or architecture promotion. A new grant must
come from the current user, not from a stored `running` or authorization label.

Do not bulk-retrofit historical frontmatter. Old status/implementation fields
remain source-time labels; a current state may point to the exact preserved
protocol without creating a retroactive preregistration. A historical entry
without current state is not automatically a completed negative or resumable
work. Keep planned, technically partial, unexecuted, invalid and scientific
negative evidence distinguishable. Do not reset a valid current grant simply
because a new record is created, or preserve an expired grant through a label.

## Research And Infrastructure Axes

Track a research unit and its supporting infrastructure independently:

| Observation | Research disposition | Infrastructure disposition |
| --- | --- | --- |
| Mechanical path is invalid before an affected contrast is durably recorded | The question is unanswered for that contrast; keep the result neutral and the evidence `none` or `partial` | Preserve the failed run receipt and route the defect to the named infrastructure owner |
| Mechanical path is accepted and the declared contrast yields a null effect | Admit bounded negative evidence if denominator, controls, and uncertainty are satisfied | Record mechanics as accepted without claiming scientific meaning |
| Model execution is affected or required raw evidence is missing | Keep the old affected contrast technically invalid and scientifically unusable | Validate the repair and require a fresh immutable affected run |
| Only derived evaluation is defective; complete raw evidence is retained | Old affected metrics remain invalid; unchanged execution may support newly derived evidence | Verify execution independence and frozen evaluation semantics, then publish a new versioned evaluation bound to the raw inputs and repaired evaluator; preserve the old artifact |
| Conditioning, population, estimand, control, or stop rule changes | Apply the semantic-delta gate and obtain any owning decision; use a new unit when the question changed | Do not disguise the semantic change as an implementation repair |

A technically invalid run is immutable provenance; it does not automatically
invalidate the research unit. Recoverable arm evidence may support only an
unaffected declared contrast. Infrastructure validation can unblock execution
but cannot complete a research unit, promote a mechanism, or rehabilitate
missing evidence.

Do not reuse execution if parser/evaluator output fed sampling, rewards,
updates, selection, or stopping affected by the defect. Such feedback requires
a fresh affected run. A repaired evaluator that changes the intended metric,
population, or claim still needs the semantic-delta gate, not just a new version.

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
- do not treat technical-invalid, unexecuted, or missing-support cases as
  scientific negatives unless the frozen protocol explicitly measures that
  failure as an outcome. Unknown/unmatched neutrality does not permit row
  exclusion, denominator changes, or automatic zero gradient; reward-neutral
  can still receive trajectory-level gradient. Valid empty/dropped outputs
  are scored under the frozen protocol, not relabeled technical failures; and
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
controls, uncertainty plan, safety gate, and complete artifact identities only
after the exploratory observation survives its control and the semantic-delta
gate is closed. Add independent review when a named unresolved risk needs it
or the governing contract requires it, not automatically. Publication- or
production-grade work may then add broader replication, human annotation, stable schemas, and
operational hardening. Do not use randomization journals, exhaustive source
closures, adversarial mutation defenses, or resume machinery to make a
pre-observation exploratory unit appear ready.

After execution begins, record changes that alter scientific meaning,
comparability, scope, or artifact attribution and use a fresh run identifier.
Ordinary implementation repairs do not require a new protocol ceremony. Never
rewrite a declared scope after observing results; label partial execution as
partial evidence.

Record an ordinary implementation repair with its exact mechanical acceptance.
Reuse the unit only when its scientific question and frozen semantics are
unchanged. New execution needs a new run identifier; derived-only repair uses
a new evaluation identity under the research/infrastructure rules above. Do
not overwrite prior failed evidence or pass its old metrics off as repaired.

Before scaling a treatment, distinguish lack of signal supply, optimization
failure, proxy-to-outcome transfer failure, and failure of the intended
mechanism. Do not let the easiest available label, candidate, metric, or
intermediate representation silently redefine the research objective. Use
[Research Alignment Examples](research-alignment-examples.md) for compact
positive and negative patterns.

When closing a non-trivial unit, separate `Observed`, `Supported`, `Ruled out`,
`Unresolved`, and `Not claimed`, then name the next discriminator.

### Conditional execution checks

For sequential interventions, regenerate each later intervention from the
cold-read native state produced by the accepted earlier intervention. Never
compose a decision-bearing route from a teacher-forced, jointly sampled, or
hypothetical prefix unless that conditioning is the frozen estimand.

For mutable runners, use one production-shaped sentinel for the applicable
invariants: intervention consumption, gradient/update direction for training,
rejection/rollback where supported, receipt completion ordering, and required
cold readback. Reuse unchanged accepted evidence; do not add rollback machinery
to a static path merely for this checklist. If an execution invariant fails,
repair the instrument and restart affected execution from the immutable parent,
not as an additional scientific arm.

## Independent Advanced-Model Review Gate

An **independent advanced-model reviewer** is a capable model or agent outside
the primary implementation and internal-audit chain. Use
`native-subagents-guidance` for task-shaped routing; the provider and effort
are not part of the scientific contract. The reviewer supplies criticism and advice.
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
4. Respect the phase/target/risk review budget across all skills. Use multiple
   reviewers only when separately authorized; give them the same core packet
   independently without revealing another reviewer's reasoning or verdict.
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
   launch decision, or next discriminator. Update the owning question/current
   context only if its meaning changes; update durable project memory only on
   an explicit user request.

## Closeout Consistency Gate

Close a unit from the evidence owner outward:

1. accept and preserve the result plus its immutable evidence receipt;
2. update the unit's `state.json`, not a status field inside frozen `unit.md`;
3. update the question page, story and `research/index.md` only when their respective
   belief, research trajectory or frontier/user-boundary meaning changes;
4. maintain catalog references without copying result ledgers or volatile counts;
5. use a handoff only for a real transfer, integrate its delta, then archive it.
   Durable project-memory changes require an explicit user request.

These surfaces may summarize the same decision, but they must not become
independent authorities. Keep one current frontier, link to the owning result,
mark superseded routes explicitly, and resolve contradictions before handoff.

A live route means any `current`, `next`, `start here`, fresh-session, or
minimum-reading-path pointer in the research entry, question page, or explicitly
authorized project memory. Its target must be a tracked owning state/result/question,
research index, current doc, or stable spec. Handoffs, standalone
agent-review or audit outputs, reviewer packets, transcripts, memory notes,
scratch files, and temporary artifacts may be cited only as provenance; they
must never own the live route. A handoff is consumed transport, not a permanent
route authority.

## Artifact Root

The research checkout's `docs/OUTPUT_STORAGE_POLICY.md` owns the source/artifact
boundary. Output roots hold execution artifacts, not loose Python, shell scripts,
Markdown, environments or vendor checkouts. Capture source bytes outside outputs
with the maintained source-provenance operation, and record its returned path.
Use an explicit hash-bound archive reader for historical evidence, never as a
fallback to satisfy a new run's current-source checks.

Use:

```text
outputs/research/<program>/<unit-id>/<run-id>/
```

Here `<program>` is the existing artifact namespace, `<unit-id>` is the immutable
research-unit identifier, and `<run-id>` is one immutable execution identifier.
Flattening `research/` does not rename existing output roots, run IDs or sealed
artifact references. This artifact namespace is not an active knowledge wrapper.

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

Before reclaiming an artifact root or deleting a producer script, search
`tests/` alongside `research/`, `memories/`, `docs/`, and every live probe
worktree. Real-fixture tests bind roots, and sealed compatibility receipts
bind source files — test modules included — by SHA-256. A file named in a
receipt binding is load-bearing bytes regardless of its test or doc status;
splitting, moving, or deleting it fails the bound consumer closed.

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
- Parse declared new metadata, especially current state JSON; do not impose
  a new lifecycle schema on frozen historical frontmatter.
- Fail review on any unexplained local abbreviation, arm code, hypothesis code,
  metric symbol, or coined name.
- From the verified research-probes checkout, run
  `python -B scripts/research/check_research_knowledge.py check` for flat layout,
  live links, catalog/state and preserved-source/Git identities, followed by
  `python -B -m unittest discover -s tests/research -p 'test_research_knowledge.py'`.
  The obsolete decision-graph checker is retired, not a second acceptance gate.
- For changed source consumers, verify their actual CPU data-read/output behavior,
  including exclusion identities and fail-closed missing inputs. Keep historical
  source recovery separate from runnable original-context replay.
- Check current-state/result consistency, including accepted-but-incomplete or
  paused outcomes. Historical snapshot status is not a competing live status;
  old next-step language must not bypass the current state and user boundary.
- Verify referenced artifact receipts/hashes when available.
- Run `git diff --check` on touched files.

Do not retrofit all historical units to a new schema. Knowledge integrity
checks target live references and source conservation, not a uniform metadata
vocabulary or new scientific verdict for every archived protocol.
