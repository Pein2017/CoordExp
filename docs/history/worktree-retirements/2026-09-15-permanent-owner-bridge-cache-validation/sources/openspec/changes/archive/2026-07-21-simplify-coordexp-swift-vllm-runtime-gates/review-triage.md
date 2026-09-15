## Review Fixed Point

- Branch: `coordexp-swift`
- Review scope: active vLLM runtime-gate change and its implementation files
- Review lanes: independent contract auditor and upstream/runtime tracer
- Initial verdict: HOLD with no P0 and overlapping P1 findings

## Accepted Findings

### P1: Execution receipt did not prove B/C application

Accepted and fixed. Materialization now validates adapter merge and embedding-
delta fold evidence against the configured source identities before atomic
publication, including one delta addition and tied input/output storage. Cache
hits revalidate the same evidence.

### P1: Rank-local live-decode evidence broke strict merge

Accepted and fixed. Semantic session normalization excludes rank-local process,
CUDA, live, and cleanup observations. Merge requires each vLLM rank to complete
live decode and cleanup, rejects
semantic setting drift, and retains one observation per rank.

### P1: Merged observations inferred rank from shard order

Accepted and fixed during fixed-point review. The merge now carries each
validated shard's declared rank together with its backend session. Reversed
shard order preserves the correct PID, CUDA, live-decode, and cleanup ownership
instead of inventing ranks with `enumerate()`.

### P1: Cleanup and process/CUDA evidence were incomplete

Accepted and fixed. Session preflight records process and CUDA binding. Every
owned engine has an explicit cleanup scope; missing shutdown support or cleanup
failure is fatal and recorded. Pipeline metadata refreshes after context exit so
successful and failed cleanup evidence reaches terminal artifacts.

### P1: Unknown versions could overclaim raw-model likelihood

Accepted and narrowed. Unknown versions may attempt policy-only execution, but
raw tracing fails unless version-specific ordering evidence exists. Known-
working vLLM `0.14.1` retains the live replay path.

### P2: Data path ownership error was bypassed

Accepted because the fix was small and correctness-aligned. The config boundary
now validates expected path kinds before CUDA discovery or JSONL loading.

### P2: Raw replay omitted native request-id comparison

Accepted because the active contract explicitly requires it. Replay now checks
the generation and replay native request ids in addition to semantic request,
prompt, token, stop, and likelihood alignment.

## Not A Code Finding

The dirty worktree also contains parallel training and pack-cache work outside
this change. Those files predated and remained outside this implementation
scope; they were preserved and are not evidence for or against this change.

The stable specs retain the previous contract until the normal OpenSpec sync or
archive lifecycle. The active delta is the authority for this unarchived change;
stable specs will not be overwritten ad hoc during implementation review.

## Verification

- Focused runtime/execution/merge/pipeline tests: 142 passed.
- Complete inference plus detection-evaluator suite: 432 passed.
- Current-source two-GPU smoke completed at
  `outputs/coordexp_swift/infer/smoke/qwen3-vl-2b-step4887-vllm-parity-smoke-20260721T042408Z/`
  with raw likelihood available and merged status
  `passed_all_rank_live_decode_and_cleanup`. Its rank evidence binds distinct
  PIDs, CUDA tokens `0` and `1`, assigned first rows, and all three cleanup
  scopes to declared ranks 0 and 1.
- Strict OpenSpec validation is rerun after the convergence review.

## Final Verdict

Both independent fixed-point lanes returned GO after inspecting the current
two-rank artifact and current source hashes. No unresolved P0/P1 findings
remain. The change is implementation-complete and ready for the normal
OpenSpec sync/archive decision; it has not been archived implicitly.
