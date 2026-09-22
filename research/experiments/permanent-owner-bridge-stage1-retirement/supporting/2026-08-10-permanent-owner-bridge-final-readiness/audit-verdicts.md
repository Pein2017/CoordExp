# Permanent owner bridge frozen-tree audit verdicts

## Fixed target

- Audited commit: `ffbb3c8b8634621b722c1a565cc4f8d766d7a405`
- Worktree: `/data/CoordExp/.worktrees/permanent-owner-bridge`
- Git state observed by every closure auditor: clean
- Executable drift boundary: `git diff 80ded9ad2..ffbb3c8 -- src/ scripts/`
  is empty. The only test change after the W1/W8 execution commit is
  `082e474d1`, which raises the parent budget of a CPU/Gloo negative-control
  test from 30 to 60 seconds while leaving its child process-group timeout at
  20 seconds.

This page retains the task 4.12 closure decisions. It is a transport record of
auditor verdicts, not a replacement for the executable receipts they checked.

## Independent verdicts

### Sol engineering-standards closure

- Verdict: `PASS`
- P0: 0
- P1: 0
- Decision: task 4.12 engineering half passes; engineering permits entry to
  task 4.13 but does not authorize task 4.14 or a GPU launch.
- Decisive checks: tracked readiness receipt/JUnit; JUnit SHA/count/skip and
  exact argv/version match; clean tree; no `src/` or `scripts/` drift; launch
  guard remains dry by default and fail-closed.
- Residual P2: after an unsafe synchronized-slot veto, the next tiny-Gloo
  window has a known rank-symmetric DDP reducer-lifecycle impurity. The current
  fixture preserves values and does not deadlock. Treat the first post-unsafe
  production heartbeat as an operations watch point; this is not a launch
  blocker.

### Sol intent-contract closure

- Verdict: `PASS`
- P0: 0
- P1: 0
- Decision: task 4.12 intent-contract half passes; entry to task 4.13 passes,
  while task 4.13 itself remains incomplete.
- Decisive checks: 2001 canonical tests reconcile to 2000 pass plus one
  explicit CUDA-hidden skip; the receipt/JUnit SHA, command, `pytest.ini`
  scope and versions agree; OpenSpec validates strictly; historical and
  launch-bearing materializations are distinguished; algorithm source hashes
  still match.

### Opus-xhigh independent closure

- Verdict: `PASS_ENTER_4.13`
- P0: 0
- P1: 0
- Decision: both earlier evidence conditions are closed; enter task 4.13 but
  do not infer authorization for task 4.14.
- Decisive checks: reproduced the CUDA-hidden collection count of 2001 and
  explained the CUDA-visible 2002 contrast; independently parsed all 2001
  JUnit cases; re-ran the import/residue checks; verified all 23 algorithm
  source hashes, both accepted `80ded9a` materializations, and the doc-only
  closure commit scope.
- Residual P2: the algorithm receipt normalizes Torch as `2.9.1`, while the
  readiness receipt records the package build string `2.9.1+cu128`; this is a
  formatting difference, not a package-version mismatch.

## Shared claim boundary

All three closure reviews support frozen-tree software, mechanical smoke,
checkpoint/reload, and bounded lifecycle readiness only. They do not establish
model quality, production GPU ownership, a production launch, or a production
optimizer heartbeat. Tasks 4.13, 4.14, and 4.15 retain those authorities in
order.
