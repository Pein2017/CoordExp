# Verification

Implementation resides in the isolated `vllm-rollout-policy` worktree based on
`249b9b0`. The canonical frozen A/B source and its released backend are unchanged.

## CPU boundary accepted

- Shared arithmetic and replay: 63 checks passed (exit 0). Original policy and
  replay baselines passed before the move. Mutations that detach factors, change
  effective-addition precision, cast factors early, or add decode scalar reads
  each failed their relevant check. The moved placeholder helper and retained
  experiment diagnostics are byte-identical.
- Resident backend: 51 checks passed (exit 0), with two real-checkpoint/CUDA cases
  deliberately excluded from CPU selection. A pre-change non-coordinate action
  demonstrated the duplicate-channel bug: returned raw log probability
  -0.551445 versus required -2.169846; the corrected consumer passes.
- Trace layer: 31 checks passed (exit 0), including complete unique action
  positions, request reordering, discarded prefill, async suffix limits, PAD/EOS,
  snapshot failures, trace-off state, and exception cleanup.
- One focused Astra review checked the installed vLLM 0.29 sampler/runner seam.
  It found a blocking per-step host-to-device row-index construction. A strengthened
  existing check failed on that implementation, and device-resident slicing
  passed. Lead inspection confirmed removal of the offending construction and
  preserved row/action selection. No other concrete mapping/channel blocker was found.
- Qualification entry and existing device-admission consumer: 13 checks passed
  (exit 0). The real two-rank Gloo entry with substituted model computation uses
  uneven 2/1 image ownership and matches an independent single-process global
  image-mean gradient reference. Failure records, child settlement, and distributed
  cleanup are exercised. No CPU test loads a real research model.

Raw CPU receipts are under this worktree's
`outputs/runtime-optimization/unify-resident-rollout-policy/`:
`shared-policy/result.json`, `backend/receipt.json`, and `trace/receipt.json`.
They retain actual commands, exits, original failures, and bounded repaired checks.
The entry check log is `qualification-cpu/pytest-final.log`.

## Remaining boundary

The real two-rank qualification in design.md remains pending. CPU acceptance does
not establish native async scheduling, measured acceleration, full-study capacity,
or scientific equivalence. Final standards/contract acceptance and safe integration
will use that saved consumer evidence without rerunning unchanged package checks.
