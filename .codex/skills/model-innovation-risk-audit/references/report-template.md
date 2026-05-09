# Model Innovation Risk Audit Report Template

Use this structure for the final audit report.

```markdown
# Model Innovation Risk Audit Report

## Scope

- Innovation:
- Branch/worktree:
- Configs:
- Checkpoints:
- Dataset:
- Audit mode: read-only / patch-planning / implementation
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

### P3

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
