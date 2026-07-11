## Audit Report Template (Read-Only)

### Scope
- Target: `<paths / change id / feature>`
- Mode: `spec/design review` | `implementation vs spec audit` | `regression risk audit` | `diff/code review`
- Intent: `<user brief / OpenSpec change / stable spec / docs contract / unavailable>`
- Fixed point: `<base ref / merge base / staged / unstaged / not applicable>`
- Constraints: `<no-network / time budget / must-run tests / do-not-run tests>`

### Safety Snapshot
- `git status --porcelain`: `<clean | dirty (list files)>`
- Environment assumptions: `<conda env, python version, GPU/no-GPU>`

### Findings (Ranked)

For `diff/code review`, keep two subsections—`Engineering Standards` and
`Intent And Contract`—and rank P0/P1/P2 within each. For other modes, use the
single severity sequence below.

#### P0 (Correctness / Data Corruption / Eval Invalidity)
- Finding:
- Evidence:
- Why it matters:
- Suggested fix direction (for implementer):
- How to verify:

#### P1 (Likely Bug / Reproducibility / Contract Drift)
- Finding:
- Evidence:
- Why it matters:
- Suggested fix direction (for implementer):
- How to verify:

#### P2 (Maintainability / Footguns / Coverage Gaps)
- Finding:
- Evidence:
- Why it matters:
- Suggested fix direction (for implementer):
- How to verify:

### Confirmed OK / Ruled Out
- `<short bullets with evidence>`

For `diff/code review`, identify the axis each confirmed check supports.

### Verification Steps (Commands)
- `<exact commands; prefer narrow tests first>`

### Open Questions (Minimize)
- `<1-3 questions that unblock the implementer>`

### Verdict
- `<approve | hold | reject | probe required | needs user decision>`
- Residual risk: `<what remains unverified>`
