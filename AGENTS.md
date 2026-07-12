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
- When subagents are allowed or requested, dispatch independent lanes instead of stacking broad raw context in one thread. Give each lane scope, permissions, and a stop condition; the parent agent must synthesize, remove duplication, and decide.

## Execution Harness

- Identify the success criterion and smallest proof before editing. Use a brief, verification-linked plan only for multi-step work.
- Create a persistent/self-driven goal only for explicitly long-running or multi-turn work; bound it and give it a concrete stop condition.
- Verify cheap assumptions locally. Ask only when a wrong interpretation would be costly; otherwise use the conservative repo-local default.
- Trace every changed line to the request, evidence, a failing check, a contract, or cleanup caused by the change. Classify material findings as `fix`, `narrow`, `drop`, `probe`, or `needs user decision`; the user owns choices that change research meaning, compatibility, cost, destructive scope, or publication.

## Judgment Taste

- Prefer concise, scalable, readable designs over broad new surfaces. Add knobs, abstractions, workflows, or interfaces only when they protect correctness or remove real complexity.
- For research mechanisms, make semantics explicit, monitorable, numerically stable, and compatible with the existing flow before expanding scope.
- Give direct verdicts when asked to compare, rank, approve, or decide. Tie the verdict to the requested axis and the concrete evidence.
- Prefer uncomfortable but specific findings over defending prior decisions. Separate symptom, root cause, uncertainty, and current-vs-historical status.
- Keep shared guidance compact and operational. Avoid generic tutorials, stale one-off details, and duplicated policy layers.

## Authority

- Use canonical docs for current behavior and workflows, starting with `docs/AGENT_INDEX.md` and `docs/catalog.yaml`.
- Use stable specs only for compatibility-sensitive contracts. Use active change artifacts only when the user or current task puts that change in scope.
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
- Put new investigations, interpretations, negative results, and durable
  research context in `research/`. Use `docs/history/` for raw provenance
  snapshots. Treat `progress/` as a legacy/deprecated archive only: read it only
  when explicitly reconstructing old evidence, migrate useful material to
  `research/`, and do not create new `progress/` records.

## Subagent context inheritance

For every V2 `spawn_agent` call, set `fork_turns` explicitly.

- `none`: self-contained discovery, artifact lookup, narrow probes.
- `1`-`3`: tasks needing only recent hypotheses or decisions.
- `all`: full-history synthesis or tasks that explicitly depend on the entire discussion.

Never omit `fork_turns`, because the runtime defaults an omitted V2 value to `all`. Explain the selected value briefly before spawning.

## Reporting

- Lead with the verdict and crucial evidence. Default to a short answer; expand only when requested or needed to support a decision. Explain unfamiliar concepts with one concrete example, demo, or artifact before adding abstract detail.
- Reviews lead with severity-ranked findings and concrete handles. If there are no findings, say so and name residual risk or skipped checks.
- Handoffs should include objective, current state, exact paths, commands, evidence scope, risks, and continuation seeds.
- For implementation or docs changes, report the outcome and any material verification, skipped checks, or residual risks. Keep the shape concise; do not force a fixed summary template.
