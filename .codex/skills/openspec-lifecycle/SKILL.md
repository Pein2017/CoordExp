---
name: openspec-lifecycle
description: Use when the user explicitly names OpenSpec, an active OpenSpec change, or a stable compatibility-sensitive CoordExp contract lifecycle action.
---

# OpenSpec Lifecycle

Use this single mode selector for CoordExp OpenSpec work. OpenSpec is downgraded governance here: use it only for stable compatibility-sensitive contracts such as training/eval behavior, config schemas, loss semantics, artifact names, or normative metric semantics. Do not use it for ordinary experiment planning, branch checklists, or implementation management.

Requires the `openspec` CLI.

## Mode Selector

- `explore`: reason about whether OpenSpec is appropriate; do not create artifacts unless the user asks.
- `new`: start a change and stop after first artifact instructions.
- `fast-forward`: create the complete apply-ready artifact set in one pass when explicitly requested.
- `continue`: create the next ready artifact for an existing change.
- `apply`: implement tasks from an existing in-scope change.
- `verify`: check workflow/artifact completeness before archive; use `audit-review` for severity-ranked correctness findings.
- `sync-specs`: apply delta specs into stable specs without archiving.
- `archive`: archive one completed change with native CLI support.
- `bulk-archive`: archive multiple changes only after explicit selection, conflict review, and confirmation.
- `onboard`: teach the workflow through a small compatibility-sensitive CoordExp example.

## Common Preflight

1. Confirm OpenSpec is explicitly in scope or the requested change is stable and compatibility-sensitive.
2. Select the change; ask if ambiguous.
3. For an existing change, run:
   ```bash
   openspec status --change "<name>" --json
   ```
4. Read every context/dependency file named by `openspec instructions ... --json`.
5. Validate strictly when specs changed:
   ```bash
   openspec validate --strict
   ```

## Mode Flows

`explore`:
- Ground the question in current docs/code/artifacts before theorizing.
- Separate hypothesis, contract requirement, implementation plan, measured result, and durable interpretation.
- End with the smallest next action: create/update artifact, use a repo-local plan, run a probe, or leave as discussion.

`new`:
```bash
openspec new change "<name>"
openspec status --change "<name>"
openspec instructions <first-ready-artifact> --change "<name>"
```
Stop after showing first artifact instructions. Do not write proposal/spec/design/tasks until the user asks to continue or fast-forward.

`fast-forward`:
- Create or reuse the change.
- For each ready artifact, run `openspec instructions <artifact-id> --change "<name>" --json`, read dependencies, and write only the requested `outputPath`.
- Re-run status after each artifact and continue until all apply-required artifacts are done.

`continue`:
- Run status, pick the first ready artifact, fetch instructions, fill `outputPath`, and stop after one artifact.
- Do not mark implementation tasks complete.

`apply`:
- Run `openspec instructions apply --change "<name>" --json`.
- Read every `contextFiles` path before editing.
- Implement pending tasks in order unless blocked by a discovered design issue.
- Update task checkboxes only for completed tasks.
- Run the smallest validation named by artifacts or repo docs.

`verify`:
- Read context files, tasks, delta specs, and design if present.
- Check task completion, implementation evidence for each requirement/scenario, contract tests/smokes, and docs/spec/artifact consistency.
- Report: `Findings`, `Confirmed OK`, `Required Before Archive`, `Recommended Follow-Ups`, `Validation Run`, `Residual Risk`.

`sync-specs`:
- Find delta specs under `openspec/changes/<name>/specs/*/spec.md`.
- Read each delta and current main spec before editing.
- Apply only named `ADDED`, `MODIFIED`, `REMOVED`, or `RENAMED` requirement/scenario content.
- Use this only when syncing without archive; ordinary archive-time spec updates belong to native `openspec archive`.

`archive`:
- Check task completion and spec-sync decision.
- Prefer native archive:
  ```bash
  openspec archive "<name>"
  ```
- Use `--skip-specs` only when the explicit decision is to archive without updating stable specs.
- Report archive result, sync decision, validation, incomplete-work warnings, and dirty state.

`bulk-archive`:
- Run `openspec list --json`; never auto-select all active changes.
- For selected changes, gather status, task completion, delta specs, and capability conflicts.
- Present a compact table and ask for confirmation.
- Archive each confirmed change with `openspec archive "<name>"`, stopping on first failure and reporting what already changed.

`onboard`:
- Run `openspec status --json`.
- Explain that OpenSpec is for stable contracts, not ordinary planning.
- Pick a tiny compatibility-sensitive example and walk through new -> proposal -> delta spec -> design if needed -> tasks -> apply -> verify -> archive.

## CoordExp Guardrails

- Do not encode future benchmark gates as current requirements.
- Do not sync experiment notes, implementation checklists, or unvalidated benchmark claims into stable specs.
- Keep docs/progress/repo artifacts as the source of truth for evidence and interpretation.
- If a task contradicts current repo docs or executable behavior, stop and surface the conflict.
- Use repo-local plans for ordinary implementation management.
