---
name: openspec-new-change
description: Use when the user explicitly wants to start an OpenSpec change for a stable, compatibility-sensitive CoordExp contract.
---

# OpenSpec New Change

Requires `openspec` CLI.

OpenSpec is downgraded governance in CoordExp. Use it for stable contracts such as config schemas, training/eval behavior, loss semantics, artifact names, or normative metric semantics. Do not start OpenSpec for ordinary experiment planning, branch checklists, or implementation management; use repo-local plans/docs instead.

## Flow

1. Confirm the requested scope is OpenSpec-worthy; otherwise say which repo-local surface should own it.
2. Derive or accept a kebab-case change name.
3. Use the default schema unless the user explicitly names another one.
4. Run:
   ```bash
   openspec new change "<name>"
   openspec status --change "<name>"
   openspec instructions <first-ready-artifact> --change "<name>"
   ```
5. Stop after showing the first artifact instructions. Do not write proposal/spec/design/tasks until the user asks to continue or fast-forward.

## Guardrails

- Do not create future research gates that require later benchmark evidence.
- Keep docs/progress/repo artifacts as the source of truth for evidence and interpretation.
- If a change already exists, route to `openspec-continue-change`.
