## Audit Report Template (Read-Only)

### Scope
- Target: `<paths / change id / feature>`
- Intent: `spec/design review` | `implementation vs spec audit` | `regression risk audit`
- Constraints: `<no-network / time budget / must-run tests / do-not-run tests>`

### Safety Snapshot
- `git status --porcelain`: `<clean | dirty (list files)>`
- Environment assumptions: `<conda env, python version, GPU/no-GPU>`

### Findings (Ranked)

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

### Verification Steps (Commands)
- `<exact commands; prefer narrow tests first>`

### Open Questions (Minimize)
- `<1-3 questions that unblock the implementer>`

