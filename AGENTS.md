# Agent Guide - CoordExp

> Shared, agent-agnostic principles for CoordExp. Keep surface-specific routes,
> command recipes, and one-off research details in `docs/`, `.codex/skills/`,
> memory, or the active user prompt. This file is only what every agent should
> know before starting.

## Operating Posture

- User request > nested `AGENTS.md` > this guide > personal/global defaults.
- Start from the user-named evidence: path, worktree, artifact, config, spec, diff, run, or question. Inspect before explaining or editing.
- Make the smallest reversible change that handles the request. Preserve unrelated work; dirty worktrees and parallel edits are expected.
- Ask only when the choice changes research meaning, is costly/destructive, publishes externally, touches secrets, or risks irreversible compatibility.
- In this checkout, Python checks normally run in the conda environment `ms`; use it explicitly if the shell is not already there.
- When subagents are allowed or requested, dispatch independent lanes instead of stacking broad raw context in one thread. Give each lane scope, permissions, and a stop condition; the parent agent must synthesize, remove duplication, and decide.

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
- Put new investigations and interpretations in research space, historical evidence in history/progress space, and only stable current behavior in docs.

## Reporting

- Reviews lead with severity-ranked findings and concrete handles. If there are no findings, say so and name residual risk or skipped checks.
- Handoffs should include objective, current state, exact paths, commands, evidence scope, risks, and continuation seeds.
- Final summaries should name changed files, verification run, skipped checks, and residual risks without restating generic process.
