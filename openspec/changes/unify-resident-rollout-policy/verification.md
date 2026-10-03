# Verification

Implementation resides in the isolated `vllm-rollout-policy` worktree based on
`249b9b0`. The canonical frozen A/B source and its released backend are unchanged.

## Original CPU candidate and its limits

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
  preserved row/action selection. That review checked the older runner's source;
  the native attempt below demonstrated that it did not match the selected runtime.
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

## First native attempt: failed before resident generation

Commit `2eb8a44c2802852cfe241602b185dcaa91d230b9` was released once through
`native-release-01.json`. The actual command exited 1 because the active V2
sampler has no `register_forward_hook`. Both ranks saved the original exception.
Six HF requests / 320 emitted actions completed; no vLLM action, replay,
backward, optimizer update, or refresh occurred. The observed HF acquisition
critical path was 47.306 seconds. This is not a completed speed comparison.

Both resident children exited normally, both distributed groups were destroyed,
and the execution owner confirmed all observed owned processes absent. Evidence:
`qualification-native-01-failure-summary.json`,
`qualification-native-01-command-exit.json`, and the full adjacent log. Preserve
these failed-attempt costs and receipts. The V2 repair reopens trace/backend
acceptance; it requires a new source/output binding before the same failed
overall qualification is rerun.

## Repaired V2 candidate

The repaired parent boundary rejects a mismatched begin snapshot, reordered
request IDs, or missing/malformed fully qualified runner and sampler names before
generation. Nine pre-change counterexamples failed; all 32 focused backend checks
pass after repair (exits 1/0). The lead inspected the ACK consumer and confirmed
the runtime names are retained in `paired_trace` receipts. Evidence:
`backend/begin-ack-receipt.json`. This CPU evidence does not qualify the V2 trace
implementation or the native runtime.

The replacement observes V2's plain callable sampler, returning its original
output after exactly one native call. It binds live request slots at admission,
snapshots GPU action positions and `num_sampled`, discards incomplete prefill
before action validation, and restores the original sampler on every abort or
finalization. Final compact counts distinguish retained rows, emitted actions,
discarded prefill, budget overshoot, and verified async post-EOS work.

The final trace suite passes 46 checks (exit 0). Both concrete failures are
preserved: the original plain-sampler hook failure and a faithful V2 fixture
rejected by the original decode-width guard. One focused Astra pass found that
the installed runner passes total `decode_query_len` into the misleadingly named
`sampler.num_speculative_tokens`; ordinary non-speculative decoding has value 1.
The corrected guard requires 1 and retains the no-speculation, one-logit-row,
TP=1/PP=1, and no-sharding bounds. The lead checked this exact installed-source
counterexample and its repair. No other blocking mapping, cleanup, per-step
synchronization, or storage counterexample was found. Evidence:
`trace/v2-repair/receipt.json` and its original RED/GREEN logs.

The parent now preserves exact `captured_rows` from that finalizer. Its existing
consumer check failed before the field was carried and passed after repair;
the fixture distinguishes emitted counts [2,3] from retained counts [3,4].
Evidence: `backend/captured-rows-receipt.json`. Unchanged shared-policy and
qualification-entry checks were reused rather than repeated. The lead accepts
the repaired CPU candidate; native acceptance remains pending.

## Remaining boundary

The repaired real two-rank qualification in design.md remains pending. CPU checks do
not establish native async scheduling, measured acceleration, full-study capacity,
or scientific equivalence. Final standards/contract acceptance and safe integration
will use that saved consumer evidence without rerunning unchanged package checks.
