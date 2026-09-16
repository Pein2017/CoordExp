---
name: research-flow
description: Lead or close a bounded CoordExp research investigation while preserving one active estimand and separating scientific, technical, and archival status.
---

# Research Flow

Use this skill when the durable product is **research knowledge** rather than
current-behavior documentation, production code, or an ordinary audit.

## Agent catch-up and knowledge maintenance

Resolve the registered research base independently of the Skill's installation
checkout. The expected base is `research-probes` at
`/data/CoordExp/.worktrees/research-probes`; verify the runtime identity/path.
Read its `research/index.md` and `research/CONVENTIONS.md` through that Project,
not by copying its research tree into the checkout hosting this Skill.
The convention is the single layout/maintenance owner; this Skill routes to it.

All primary readers are agents. Fast catch-up is `research/index.md` plus the
current unit state/result; deep catch-up adds `story.md`, relevant question pages
and decisive sources. Before recommending a new direction or creating a unit,
search `research/experiments/catalog.jsonl` and read the closest decisive results,
including successful baselines and limiting counterexamples. Give one compact
predecessor recap: what is settled; what remains unresolved; what this proposal
changes; and which possible outcomes change the next decision. Check the proposal
against the user's ultimate objective, not only a convenient intermediate metric.
Name replication, scale checks and execution qualification as such; do not
present them as a new algorithmic discriminator. Reuse settled evidence unless
the new task crosses its stated boundary. If the incremental question is unclear,
resolve it in discussion before expanding preparation or compute. Keep the recap
in the existing discussion/unit rather than a second checklist or review chain;
continuation within an unchanged contract reuses it without full-history rereads.

The entire repository serves this research topic. Knowledge lives directly in
`research/`, with no topic/program wrapper. Its `index.md` combines frontier and
routing; `story.md`, `glossary.md`, `alternatives.md`, `questions/` and
`experiments/` have the roles defined once in the convention. Retire old OKF,
idea/decision/mechanism/archive buckets; tested ideas enter their question and
catalog, while genuinely unresolved sparks keep explicit reopening conditions.
Do not recreate old-path aliases or an empty placeholder category.
Frozen protocol, current `state.json`,
accepted results, question-oriented synthesis and historical transport have
separate roles. A historical `running` field or old compute grant is not live
state or permission. Handoffs are consumed transport, not permanent frontiers.
Do not append every run to the current context or duplicate full result tables
across routers. Archive original evidence byte-faithfully before compression;
check path/hash-bound consumers before relocating code or records.

## Authority

- Current operator guidance owns current behavior.
- Stable specifications own compatibility-sensitive contracts.
- The active research record owns ideas, investigations, mechanisms,
  decisions, negative results, and continuation context.
- The research unit owns its question, contrast, population, conditioning,
  estimand, evidence, claim, and stop rule. A named infrastructure owner owns
  reusable runtime, instrumentation, identity, persistence, and recovery
  behavior. Neither owner's acceptance closes the other.
- Historical records are provenance; legacy progress archives receive no new
  records.
- Research worktree lifecycle, return or migration of knowledge and maintained
  code, and retirement are owned by the current research base's
  `docs/BRANCH_AND_WORKTREE_POLICY.md`. Resolve that owner before acting; follow
  its preservation distinctions rather than imposing a blanket records-only
  return. Do not restate lifecycle rules here or substitute the root production
  policy for the research-base policy.
- Reviews, handoffs, packets, and memory route to owners but do not become
  authority.

Use `audit-review` for correctness findings, `model-diagnosis` for abnormal
behavior, the owning code/config/runtime path for a reproducible mechanical
failure, `coordexp-infer-eval-workflow` for run operations, and `git-hygiene`
for repository publication.

These skills answer different questions, not successive mandatory approvals.
Reuse evidence closing the same phase/target/risk; changing skills does not
restart the review budget. When delegation is worthwhile, use
`native-agent-team-guidance` for roles, model/effort, and message flow. Package
owners integrate their work; the lead verifies the decision-bearing boundary
rather than repeating every internal check. Model capability never substitutes
for evidence or grants research authority.

## Active-contract gate

Before executing a probe or training step, freeze one current sentence:

```text
From anchor A, does intervention X change decision outcome Y under gate Z?
```

Bind it to immutable checkpoint/config/input identities, intervention boundary
and provenance, evaluation/owner ledger, acceptance level, attempt budget, and
stop rule. Existing unit/config/receipts may hold this contract; do not create
a duplicate packet. Bind prefix identity when conditioning or replay depends
on it. Exact-token identity is a replay/provenance diagnostic unless the unit
explicitly makes exact replay the estimand.

An auxiliary proxy, null, mechanism diagnostic, or gradient screen is
decision-owning only when the active contract explicitly names it and its stop
rule is user-authorized. Otherwise run it after or alongside the frozen primary
contrast, never on its critical path: failure, infeasibility, or ambiguity closes
only that diagnostic branch. This does not weaken the input, identity,
mechanical-validity, safety, or evidence-interpretability gates required by the
primary contrast.

A change to checkpoint family or ordering, conditioning surface, intervention
sequence, acceptance/debt policy, architecture, claim, critical path, or stop
rule is a phase reset. Close or detach the old contract and create a fresh unit
or continuation packet before execution; do not let old terms silently acquire
new meanings. Start a fresh task when the lead must reconstruct more than one
prior iteration from conversation rather than from the packet.

## Work

1. **Bound the source and product.**
   - Record exact roots, worktrees, date window, globs, artifacts, and whether
     the result is raw intake, migration, synthesis, a research unit, or a
     decision packet.
   - Classify the requested product as research knowledge, infrastructure
     behavior, or a linked pair. Give a linked pair two explicit owners rather
     than one blended status.
   - Complete when scope cannot silently expand.

2. **Preserve provenance before interpretation.**
   - Keep source path, checkout, content identity, and divergence visible.
   - For cross-worktree unions, distinguish identical content, new paths, and
     same-path divergent content before cleanup.
   - Complete when every retained claim can be traced to a source.

3. **Synthesize into an owning surface.**
   - Build a reading path rather than mirroring raw notes.
   - Separate planned protocol, executed evidence, interpretation, decision,
     mechanism promotion, and implementation authorization.
   - For executed work, state scientific disposition separately from
     mechanical validity and infrastructure follow-up.
   - Distinguish observations, hypotheses, and supported inferences. When
     interpreting a mechanism or choosing the next experiment, name the
     strongest remaining alternative and cheapest discriminating evidence;
     do not expand this into a mandatory mechanism matrix.
   - For research units, apply
     [Research Graph and Unit Contract](references/research-graph-contract.md).
   - Complete when one tracked unit/result/question owns the current statement.

4. **Resolve semantic deltas.**
   - Compare the originating question and approved route with any later change
     to cohort, predicate, estimand, control, claim, stop rule, critical path,
     or material cost.
   - Record old versus new, reason, claim/evidence/cost impact, authorization,
     and detached work.
   - Record decision-owning outcome, intervention or proxy, final evaluation,
     transfer assumption, preservation risk, and signal supply.
   - Compare natural versus forced or teacher-forced conditioning,
     precondition or admission versus conditional realization, and source
     population versus eligible, executed, and analyzable denominators.
   - Use [Research Alignment Examples](references/research-alignment-examples.md)
     only when the boundary is unclear.
   - Complete when derived constraints are either authorized or explicitly
     labeled proposals.

5. **Gate evidence on both axes.**
   - On the research axis, verify the declared question, contrast, denominator,
     conditioning, evaluation surface, and permitted claim.
   - On the infrastructure axis, verify inputs, geometry and token boundaries,
     runtime identity, intervention consumption, serialization, persistence,
     and readback required by that contrast.
   - Treat oracle, forced-prefix, teacher-forced, retrieval, and mechanics-smoke
     observations as evidence only for their exact surface. Do not promote them
     to natural behavior or trainability without an explicit transfer test.
   - Do not interpret technical-invalid, unexecuted, or missing-support cases
     as scientific negatives unless the frozen protocol explicitly measures
     that failure as an outcome. Unknown/unmatched cases have only the meaning
     assigned by the frozen protocol: neutrality does not authorize dropping
     rows, changing denominators, or forcing zero gradient. Reward-neutral is
     not necessarily gradient-neutral; valid empty/dropped model outputs may
     be real zero-score outcomes rather than technical failures.
   - Bound a technical failure to the affected evidence/contrast. For a
     derived-evaluation-only defect, complete retained raw evidence may support
     a new versioned evaluation after verifying execution was unaffected and
     the repaired evaluator preserves the frozen semantics. Preserve the old
     invalid evaluation; missing raw evidence or affected model execution
     requires a fresh affected run. Never silently rehabilitate old metrics.
     A green infrastructure check proves mechanics only.
   - Complete when every verdict names both the accepted mechanical path and
     the evidence-bearing scientific contrast, or states that one is absent.

6. **Close routing once.**
   - Close from the owning accepted receipt/result to the unit's `state.json`.
     Update a question page only when belief changes, the story only when the
     research trajectory changes, and `research/index.md` only when the frontier
     or user boundary changes. Maintain catalog paths, not another status table.
     Create a handoff only for a real transfer and archive it after integration.
     Change durable project memory only on an explicit user request.
   - Route a reusable technical deficiency to its infrastructure owner with the
     exact failure and acceptance boundary. Link it from the research unit, but
     do not move cohort, intervention, estimand, threshold, or claim ownership
     into infrastructure.
   - Promote current behavior to operator guidance or stable contracts only
     through their own authorized change.
   - Complete when no live router points at superseded transport or provenance.

   Keep one machine-readable iteration receipt authoritative. Human records
   link to it and summarize the decision; do not duplicate volatile counters or
   ledgers across multiple prose surfaces. Update broad routers at a promoted
   anchor or final closeout, not after every attempted update.

7. **Verify the product.**
   - Check tracked scope, links, source counts, lifecycle state, artifact
     attribution, nested denominators, technical-failure accounting, and that
     the verdict does not exceed its evidence.
   - Complete when the reading path is coherent and the authority boundary is
     explicit.

## Research OpenSpec Closeout

When closing, retiring, or preparing to archive a research-owned OpenSpec
change, read [Receipt-Grounded Closeout](references/openspec-closeout.md) before
changing checkboxes. Literal task conditions and immutable receipts govern
completion; archive authorization remains separate.

## Conditional Branches

- For sequential interventions or mutable runners, apply the conditional
  execution checks in [Research Graph and Unit Contract](references/research-graph-contract.md).
  Do not require prefix or rollback machinery for unrelated static work.
- For raw Markdown union collection, preserve a manifest and byte-faithful
  snapshots before synthesis.
- For supervisor or independent-model packets, make the packet
  decision-focused and keep reviewer output advisory.
- For knowledge migration, use the research base's `research/CONVENTIONS.md`:
  flat root entry, question/story synthesis, source manifest and targeted
  retrieval. Keep metadata light; do not retrofit historical protocols merely
  to satisfy a schema or introduce an app-specific wiki framework.
- For executed units, keep durable artifacts under the owning output root and
  record immutable run identity once comparison requires it.
- Before support expansion or a mechanism matrix, require CPU-discoverable
  receipt/write-read checks, one minimal real end-to-end mechanics smoke, and
  then the smallest decision-bearing scientific pilot. Do not let in-memory
  outputs or helper-only tests stand in for durable end-to-end evidence.
- For a long self-driven loop, predeclare the attempt budget and phase-reset
  triggers. Use at most one advisory pass at an unresolved semantic fork and
  at most one independent review when a named promotion/flatten risk needs it
  or the governing contract requires it. Share that budget across skills; do
  not review a changing target or run cleanup lanes against active science.
- Report launch, anomaly, decision-bearing result, and closure. Do not turn
  command liveness, buffered logs, or routine receipt reads into a second
  status stream.

## Report

State source boundary, created or updated owners, raw versus synthesized
material, scientific disposition, infrastructure disposition, authority
caveats, verification, unresolved gaps, and next continuation point.
