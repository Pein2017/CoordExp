# Agent Guide - CoordExp

> Shared, agent-agnostic principles for CoordExp. Keep surface-specific routes,
> command recipes, and one-off research details in `docs/`, `.codex/skills/`,
> memory, or the active user prompt. This file is only what every agent should
> know before starting.

## Operating Posture

- User request > nested `AGENTS.md` > this guide > personal/global defaults.
- Start from the user-named evidence: path, worktree, artifact, config, spec, diff, run, or question. Inspect before explaining or editing.
- Make the smallest reversible change that handles the request. Do not add features, knobs, cleanup, formatting, or refactors unless they are required by the request or by verification. Preserve unrelated work; dirty worktrees and parallel edits are expected.
- Ask only when the choice changes research meaning, is costly/destructive, publishes externally, touches secrets, or risks irreversible compatibility.
- In this checkout, Python checks normally run in the conda environment `ms`; use it explicitly if the shell is not already there.
- When subagents are allowed or requested, dispatch independent lanes instead
  of stacking broad raw context in one thread. Give each lane the smallest
  self-contained brief, scope, permissions, evidence handles, and a stop
  condition; the parent agent must synthesize, remove duplication, and decide.
- Prefer generic subagents with task-specific briefs. Choose the model and
  reasoning effort at dispatch time rather than relying on a fixed
  role-to-model mapping.
- Before substantial delegation, read `.codex/MODEL_ROUTING.md`. It is a
  living, evidence-based routing guide rather than an executable role map;
  monitor outcomes and revise its defaults when repeated task evidence changes
  the cost-quality boundary.
- Installed external-worker bridges such as `cc:*` are eligible peer
  delegation surfaces, not mandatory review gates. They may own bounded audit,
  investigation, implementation, or writing lanes when dynamically selected;
  the lead agent still owns research interpretation, scope changes, synthesis,
  and final judgment.
- Treat an external `--write` lane as a real writer: assign one semantic owner,
  state its bounds and verification, and protect concurrent local changes.
- Use a cost-efficient model when a command, test, artifact receipt, or other
  mechanical check can decide success. Use the strongest appropriate model
  when judgment can change scientific meaning, architecture, or a
  conclusion-critical implementation; escalate only when evidence warrants it.
- Give one current owner to each independent decision surface. Do not create
  duplicate implementation lanes or repeated general reviews merely because
  agent slots are available.
- Keep implementation ownership and independent judgment separate when that
  distinction matters. Reviewers should reconstruct the claim from the named
  code, specification, diff, tests, and artifacts instead of inheriting the
  implementer's reasoning or the parent thread's preferred conclusion.
- Preserve source reviews and audits as provenance, but record later changes as
  evidence or deltas instead of creating duplicate current authorities.

## Communication and Language

- The user may communicate in Chinese, English, or a mixture of both. Treat all
  languages as equally authoritative; do not interpret English as more formal
  or Chinese as less precise.
- Chinese is permitted for interactive user-facing discussion only. Follow the
  language of the latest user request there, while preserving technical terms
  in English. When precision, implementation semantics, or scientific meaning
  could benefit from English, prefer English or add an explicit English
  operational restatement.
- Keep code, paths, commands, configuration keys, schemas, formulas, experiment
  identifiers, plans, specifications, reviews, handoffs, and all other durable
  repository artifacts in English unless the user explicitly requests a
  file-specific exception.
- Translate Chinese research intuition into an explicit English implementation
  or experiment contract when needed. If translation could change scientific
  meaning, state the interpretation rather than silently choosing one.
- All agent and subagent instructions, briefs, intermediate reports, review
  findings, and handoffs must be written in English. Parent discussion with the
  user may remain Chinese, and the parent may return a Chinese synthesis after
  reconciling the English agent outputs.
- When Chinese discussion and an English operational contract appear to differ,
  do not silently choose either interpretation. Surface the discrepancy and
  resolve it before implementation; once resolved, record the executable
  contract in English.
- Every abbreviation, shorthand name, experimental-arm identifier, hypothesis
  identifier, and coined term in durable artifacts must include its complete
  name and operational meaning.
- Use ordinary, concrete language. Introduce an acronym, metaphor, or method
  name only when it is clearer than a short description of the actual
  comparison or behavior, and explain it in plain language at first use.

## Execution Harness

- Before changing files, identify the task type, success criterion, and smallest evidence that would prove the work is done. For tiny edits this can stay implicit, but it must still guide the change.
- Create or refine a persistent/self-driven goal only for explicit long-running or multi-turn work, and only after bounding the lane; the goal must include a concrete stop condition.
- For research implementation, outline the question, minimal execution path,
  expected owner surfaces, reused infrastructure, non-goals, first real smoke,
  and rough cost. Do not pretend that a speculative code interface can be
  frozen before runtime evidence exists.
- Once a research goal is authorized, let the lead adapt implementation,
  controls, sample scope, and promising recursive probes within the same broad
  objective. Escalate major direction changes or material critical-path cost
  using the task-local budget as a flexible judgment, not a universal cap.
- State assumptions only when they affect implementation, research meaning, cost, compatibility, or the verification path. If an assumption is cheap to verify locally, verify it instead of asking.
- If multiple meaningful interpretations exist, present the tradeoff and ask or pause only when the wrong choice would be costly; otherwise choose the conservative repo-local default and continue.
- For multi-step work, use a brief plan with a verification handle for each step. For simple work, proceed directly and keep the verification path explicit.
- Every changed line should trace to the user request, concrete evidence, a failing check, a documented contract, or cleanup caused by the current change.
- If a finding implies `fix`, `narrow`, `drop`, `probe`, or `needs user decision`, make that decision before patching through it.
- During an exploratory research slice, expand implementation only when needed
  to obtain the primary observation or protect its interpretation. Defer
  elegance, future consumers, exhaustive manifests, and speculative edge cases
  until the pilot justifies promotion.

## Experimental Project Memory

- The repository-local `memories/` feature and `project-memory` skill are
  experimental and should improve from observed use rather than be treated as
  a stable compatibility contract.
- For a non-trivial continuity-sensitive task, when `memories/config.yaml`
  exists, the main agent recalls `memories/current.md` once at task start and
  checks only relevant notes. Do not repeat full recall on every turn.
- Track whether the live goal, durable decision, evidence-backed conclusion,
  claim boundary, blocker, costly continuation state, or next action changed.
  If it did, checkpoint automatically at the next meaningful boundary or
  before handoff, compaction, stop, or final response; routine progress does
  not qualify.
- Only the main agent or an explicitly designated consolidator rewrites
  `current.md`. Verify dynamic process, artifact, branch, and worktree claims
  before acting because memory is continuity guidance, not live authority.

## Judgment Taste

- Search for the narrowest existing repository owner first, then a standard
  library or native platform mechanism, then an installed dependency, and only
  then add the minimum new implementation. This is a search order, not an
  automatic preference: preserve research semantics, validation, ordering,
  compatibility, and provenance.
- Prefer concise, scalable, readable designs over broad new surfaces. Add knobs, abstractions, workflows, or interfaces only when they protect correctness or remove real complexity.
- For research mechanisms, make semantics explicit, monitorable, numerically stable, and compatible with the existing flow before expanding scope.
- Give direct verdicts when asked to compare, rank, approve, or decide. Tie the verdict to the requested axis and the concrete evidence.
- Prefer uncomfortable but specific findings over defending prior decisions. Separate symptom, root cause, uncertainty, and current-vs-historical status.
- Keep shared guidance compact and operational. Avoid generic tutorials, stale one-off details, and duplicated policy layers.

## Authority

- Use canonical docs for current behavior and workflows, starting with `docs/AGENT_INDEX.md` and `docs/catalog.yaml`.
- Use stable specs only for compatibility-sensitive contracts. Use active change artifacts only when the user or current task puts that change in scope.
- Live research routers and current reading paths may target only tracked owning
  research units, results, decisions, compasses or indexes, current docs, and
  stable specs. Handoffs, agent-review outputs, reviewer packets, audit
  scratch, transcripts, memory notes, and temporary artifacts may be
  provenance, but never the current or next route target.
- Treat historical notes, old worktrees, memories, and research writeups as evidence or idea context, not current-behavior authority. Revalidate live files before relying on them.
- Work in the exact checkout or worktree named by the user. Do not mix facts across roots without checking the target root.

## Safety Principles

- Preserve semantic alignment end to end: data, images, coordinates, prompts, tokens, losses, metrics, configs, and artifacts must not be silently dropped, reordered, resized, reinterpreted, or compared across incompatible scopes.
- Prefer explicit config/schema contracts and fail-fast behavior over hidden compatibility. Unknown, obsolete, or removed surfaces should not quietly become defaults.
- Keep compatibility shims visibly separate from canonical behavior.
- Treat upstream/vendor/runtime boundaries as correctness boundaries. When they own behavior, verify the installed or executed semantics instead of trusting plans, receipts, mocks, or memory.
- Do not add hidden agent persistence, credentials, services, production dependencies, expensive jobs, destructive cleanup, broad git operations, or data deletion without explicit approval.

## Evidence Routine

- Diagnose behavior from the exact artifacts, files, or runs the user names before theorizing from config, docs, or memory.
- For experiments and model behavior, record the evidence scope: config, checkpoint or version, artifact root, counters, metric files, representative samples, and known limitations.
- Every code, config, data, docs-contract, or workflow change needs a verification path: test, smoke, parse, artifact/manifest check, metric check, replay, residue grep, or explicit skipped reason.
- Narrow checks first; broaden only when shared contracts or user-facing workflows changed. Label partial evidence honestly and never present it as full validation.
- Prefer a representative real smoke and compact run receipt over exhaustive
  pre-run assurance. Add stronger runtime guards only for a demonstrated
  failure mode that could change the scientific conclusion.
- Put new investigations, interpretations, negative results, and durable
  research context in `research/`. Use `docs/history/` for raw provenance
  snapshots. Treat `progress/` as a legacy/deprecated archive only: read it only
  when explicitly reconstructing old evidence, migrate useful material to
  `research/`, and do not create new `progress/` records.
  
## Subagent context inheritance
For every V2 `spawn_agent` call, set `fork_turns` explicitly.
- Prefer `none`: use it for self-contained discovery, artifact lookup, narrow
  probes, bounded implementation, and independent audit or review. Put the
  required contract and exact evidence paths in the brief rather than passing
  conversational history.
- Use `1`-`3` only when the task genuinely depends on recent hypotheses,
  approvals, or decisions that cannot be stated compactly in the brief.
- Use `all` only for explicit full-history synthesis or when reconstructing the
  discussion itself is the task. It is not the default for implementation,
  audit, or review.
- For competing audits, give reviewers the same evidence scope and do not show
  them one another's findings before they reach independent verdicts.
Never omit `fork_turns`, because the runtime defaults an omitted V2 value to
`all`. Explain the selected value briefly before spawning.

## Reporting

- Reviews lead with severity-ranked findings and concrete handles. If there are no findings, say so and name residual risk or skipped checks.
- Handoffs should include objective, current state, exact paths, commands, evidence scope, risks, and continuation seeds.
- For implementation or docs changes, report the outcome and any material verification, skipped checks, or residual risks. Keep the shape concise; do not force a fixed summary template.
