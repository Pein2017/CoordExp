# Model Innovation Risk Audit Report Template

Use only the sections needed for a standalone decision report. Omit empty
severity headings and unrequested P2 hardening; distinguish proven defects
from unproven gates rather than filling a checklist.

```markdown
# Model Innovation Risk Audit Report

## Scope

- Innovation:
- Branch/worktree:
- Configs:
- Checkpoints:
- Dataset:
- Audit mode: read-only (implementation requires separate authorization)
- Commands/probes:

## Findings

### P0

None / findings.

### P1

1. Finding title

Evidence:
Impact:
Fix direction:
Minimal tests:

### P2

1. Finding title

Evidence:
Impact:
Fix direction:
Minimal tests:

## Confirmed OK

- ...

## Decision Questions

1. ...

## Patch Recommendations

1. ...

## Unit Tests And Diagnostics

- ...

## Smoke Run Suggestions

- ...

## Residual Risks

- ...

## Verdict

- promote / hold / rerun gate / needs user decision
- Missing evidence and the smallest discriminator, if decision-relevant:
- Next owner:
- Evidence recommendation only; no authority to implement, launch, or publish.
```

## Finding Format

Each finding should be concise but evidence-backed:

```markdown
### P1: Title

Evidence:
- File/line or command output.
- Materialized config or runtime probe if relevant.

Impact:
- Explain how this can silently change training signal, decode behavior, eval validity, or reproducibility.

Fix direction:
- Smallest owner-surface patch.

Minimal tests:
- Test file and behavior.
- Expected failure before patch if implementation is requested.
```
