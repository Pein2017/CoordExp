# Agent Defaults

## Scope
- Personal/global defaults for autonomous coding agents.
- Repository and nested `AGENTS.md` files override this file for project-specific rules.
- User instructions override all instruction files.

## Operating Loop
- Inspecnts are underspecified; proceed with the smallest reversible change.
- Prefer implementation plus verift relevant files before editing.
- State assumptions when requiremeication over extended planning.
- Ask only when a choice affects research meaning, high cost, destructive cleanup, external publication, security/privacy, or irreversible compatibility.
- Keep changes scoped; do not refactor unrelated code.

## Codex Agent Operations
- Use subagents when parallel work materially helps: independent audits, subsystem exploration, disjoint implementation slices, or separate verification tracks.
- Remember the active subagent capacity is limited to 6. Close completed or no-longer-needed subagents promptly before launching more.
- For complex or long-running tasks, set an explicit `/goal` that captures the current objective, scope boundary, and stop condition before substantial execution.

## Change Safety
- Preserve unrelated user work. Inspect dirty state before staging, committing, or broad edits.
- Do not run destructive commands or delete data without explicit approval.
- Do not add production dependencies, services, credentials, or expensive long-running jobs without approval.

## Verification
- Every code, config, data, or workflow change needs a verification path: targeted test, smoke run, artifact check, metric check, replay, or explicit reason skipped.
- Run narrow checks first; broaden only when shared contracts or user-facing workflows changed.
- Before production training, deployment, or release, verify inputs, configs, caches, artifacts, metrics, and rollback/restore paths.

## Evidence
- Attach concrete handles to recommendations and changes: file paths, symbols, config keys, commands, artifact roots, metrics, or minimal I/O examples.
- Do not invent results. Label evidence scope such as `tiny`, `smoke`, `val200`, `proxy`, `partial`, or `full`.
- Update docs/specs/configs when behavior, schema, artifact names, metric semantics, entrypoints, or recommended workflows change.

## Response Format
- Report changed files, verification commands, skipped checks, residual risks, and next actions.
- Keep final responses concise unless the user asks for a detailed audit or handoff.
