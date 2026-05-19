# Recording

CoordExp has several durable surfaces. Pick the narrowest surface that matches
the resolved decision.

## Routing

- Use the active super-power plan/spec for implementation details, command
  plans, verification checklists, and branch-local handoff notes.
- Use `progress/` for measured results, diagnostics, benchmark evidence,
  empirical failures, artifact guides, and historical reasoning.
- Use `docs/` for stable current behavior, recommended workflows, entrypoints,
  artifact names, metric meaning, and operator-facing architecture. Route
  through `docs/AGENT_INDEX.md` and `docs/catalog.yaml`.
- Use `openspec/specs/` only for normative compatibility-sensitive contracts:
  training/eval behavior, config schemas, loss semantics, artifact names, or
  normative metric semantics.
- Use `openspec/changes/<active-change>/` only when an active change is
  explicitly in scope.
- Use repo configs, tests, scripts, manifests, and artifact paths for
  executable truth whenever prose would become stale.

## Record Shape

For decision notes, keep the record short:

```md
## Decision

{What is now true or planned.}

## Rationale

{Why this is the right trade-off, with rejected alternatives only when useful.}

## Consequence

{What must change, stay compatible, or be verified.}

## Evidence

- Scope: `tiny|smoke|val200|proxy|partial|full|none-yet`
- Handles: `{configs, artifacts, tests, metrics, docs, commits}`
```

For progress notes, prefer dated, evidence-first entries. Include the config,
checkpoint, artifact root, parse/drop counters, metric files, and evidence
scope before interpretation.

## Anti-Patterns

- Do not create root `CONTEXT.md` or `docs/adr/` just because an imported skill
  expects them.
- Do not turn ordinary experiment planning into OpenSpec.
- Do not move stable contracts into `progress/`.
- Do not leave decisions only in the context window when they will steer later
  implementation, evaluation, or interpretation.
