# Agent Guide - CoordExp

> Stable, agent-neutral invariants for work in this repository. Put task-specific
> procedures in skills, current routes in `docs/`, and experiment state in the
> owning research unit. Do not turn this file into a second knowledge base.

## Grounding

- User request > nested `AGENTS.md` > this guide > surface defaults.
- Start from the exact path, worktree, artifact, config, spec, diff, run, or
  question the user names. Inspect it before explaining or changing it.
- Work in the exact checkout in scope. Revalidate facts before crossing roots;
  branches, worktrees, memories, and old notes are not interchangeable.
- Make the smallest reversible change that satisfies the request and its
  verification. Preserve unrelated dirt and parallel work.
- Python checks in this checkout normally use the `ms` conda environment.

## Authority And Judgment

- When the named evidence does not reveal the current owner, search
  `docs/catalog.yaml` or `docs/AGENT_INDEX.md` for the narrowest route; do not
  load both or follow a fixed read order by default. Use stable specs only for
  compatibility-sensitive contracts and active change artifacts only when the
  task puts that change in scope.
- Treat research notes, memories, handoffs, reviews, transcripts, and historical
  worktrees as evidence or routing context, never as live authority by default.
- The user owns choices that change research meaning, compatibility, publication,
  material cost, or irreversible behavior. The agent owns discoverable facts and
  reversible implementation details.
- Ask only when the unresolved choice crosses that boundary. Otherwise inspect,
  choose the conservative repo-local interpretation, and continue.
- Give each decision or write surface one owner. Independent review informs the
  lead; it does not replace evidence or majority-vote a scientific conclusion.

## Research And Design

- Before a research implementation or costly launch, make the question,
  contrast, decision-owning outcome, strongest alternative, primary evidence,
  and stop rule explicit.
- For an exploratory slice, implement only what obtains the primary observation
  or protects its interpretation. Prefer one representative real smoke over
  production-shaped preflight.
- Let the lead adapt reversible probes and controls inside an authorized goal.
  Escalate a major direction change or material new critical-path cost.
- Keep model, data, geometry, order, prompts, tokens, objectives, metrics, and
  artifacts semantically aligned. Report proxy-to-outcome assumptions rather
  than silently promoting them.
- Treat a proposed reusable interface as provisional until runtime evidence or a
  second real consumer establishes the seam.

## Safety And Evidence

- Use explicit config and schema contracts; unknown or retired surfaces should
  fail visibly rather than become hidden defaults.
- Verify installed upstream or runtime behavior when it owns the claim. Plans,
  mocks, banners, and receipts do not substitute for executed semantics.
- Do not add credentials, hidden persistence, services, production dependencies,
  expensive jobs, destructive cleanup, broad Git operations, or publication
  without the authority required by the user request.
- Every change needs proportionate evidence: a targeted test, real smoke, parse,
  artifact or manifest check, metric check, replay, residue search, or an explicit
  reason the check was skipped.
- Narrow checks first. Label evidence scope and residual risk; do not present a
  partial check as full validation.

## Progressive Disclosure

- Use the narrowest matching skill. A skill owns its procedure; this guide does
  not restate it.
- Use `agent-routing` for substantial delegation or a material model and effort
  choice. Use task-specific briefs, bounded evidence, one semantic owner, and
  an explicit stop condition.
- When `memories/config.yaml` exists and continuity matters, use
  `project-memory`; verify dynamic state before relying on it.
- Prefer live routers, expressive scripts, schemas, tests, and artifacts over
  memorized path lists or repeated examples.

## Communication And Durable Outputs

- Follow the user's language in conversation. Keep code, paths, commands,
  configs, schemas, formulas, experiment identifiers, plans, specifications,
  reviews, and handoffs in English unless the user requests otherwise.
- If translating research intuition could change its meaning, state the English
  operational interpretation and resolve discrepancies before implementation.
- Put new investigations, interpretations, negative results, and continuation
  context in `research/`. Use `docs/history/` for raw provenance; treat
  `progress/` as a legacy archive and create no new records there.
- Reviews lead with evidence-backed findings and a decision. Handoffs route to
  owning state rather than duplicating it. Implementation reports state outcome,
  verification, skipped checks, and residual risk.
