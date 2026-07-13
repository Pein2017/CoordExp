# Recording A Resolved Decision

Write only resolved outcomes. Prefer the narrowest authoritative carrier:

- `research/`: ideas, investigations, interpretations, negative results,
  mechanisms, and continuation context;
- `docs/`: stable current behavior and recommended workflows;
- `openspec/specs/`: normative compatibility-sensitive contracts;
- `openspec/changes/<change>/`: bounded changes needing durable planning state;
- configs, tests, scripts, manifests, and artifacts: executable truth.

Treat `progress/` as deprecated provenance, not a carrier for new records. Skip
placeholder docs, generic ADRs, root `CONTEXT.md`, and duplicate summaries.

Use `probe` for cheap evidence and `build-probe` when new hooks, data, logging,
evaluator support, or feature work is required. A `build-probe` record names the
research question, evidence target, minimal build, scope, matched baseline, stop
condition, and carrier.

```md
## Decision
{Resolved outcome, rationale, and consequence.}

## Evidence
- Scope: `tiny|smoke|val200|proxy|partial|full|none-yet`
- Handles: `{configs, artifacts, tests, metrics, docs, commits}`

## Next State
{drop|narrow|probe|build-probe|implement|document|needs user decision}

## Carrier
{chat|handoff|research|docs|openspec|none}
```

Continue unless the user asks to pause or only record. Recording never implies
implementation permission.
