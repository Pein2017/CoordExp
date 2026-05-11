thread_id: 019dc79a-f919-7e82-8631-ac66e5d53b3b
updated_at: 2026-04-27T16:52:11+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/26/rollout-2026-04-26T02-25-15-019dc79a-f919-7e82-8631-ac66e5d53b3b.jsonl
cwd: /data/CoordExp
git_branch: main

# The user took over a Stage-1 MP packed-branch implementation and asked for a handoff prompt for a future Codex agent to continue work.

Rollout context: The work happened in `/data/CoordExp/.worktrees/stage1-mp-padding-free-branch-packing-spec` on a Stage-1 set-continuation packed-varlen OpenSpec/worktree. The user’s focus shifted from basic packed-runtime correctness to improving performance via **cross-sample branch packing** while preserving the exact MP objective and keeping production default as smart-batched. The user also explicitly asked for a take-over prompt for another agent at the end.

## Task 1: Build and validate packed-branch preflight / smoke infrastructure

Outcome: success

Preference signals:
- The user repeatedly emphasized that packing must be verified for **mathematical equivalence**, not just throughput, indicating future agents should default to exactness-first validation rather than performance-only smoke tests.
- The user asked for adversarial tests covering “attention mask correctness”, “token-position alignment”, “boundary token handling (continuation vs termination)”, and “FlashAttention compatibility”, indicating they want packed-runtime changes proven with structural and synthetic failure-mode tests before adoption.
- The user asked for “few-step real training” and “do NOT proceed to production training before full validation”, indicating that short, production-like smokes are a required gate, not optional confirmation.
- The user asked whether the packed path could preserve around “60~70GB” memory usage, indicating memory budgets are a concrete acceptance target for future runtime changes.

Key steps:
- Added adversarial packed-preflight tests for shared-prefix branches, prior-segment isolation, masked boundary/termination loss behavior, and token-level alignment traces.
- Added a token-level debug helper (`packed_alignment_debug_rows(...)`) so future agents can inspect per-token segment index, local/global position, text position IDs, and whether losses are enabled.
- Tightened config gates so `packed_varlen_exact` requires the packed runtime and FlashAttention flags, and reserved memory-budget fields remain rejected until they are actually enforced.
- Updated the smoke profiles to use `max_total_tokens_per_forward: 14000` and `max_segment_tokens: 14000` after the user clarified the current global max length is 14k.

Failures and how to do differently:
- The first packed preflight test run failed because the test fixture did not satisfy the new runtime gate (`branch_packing.enabled=true`), so the test was asserting the wrong failure mode. Future agents should mirror the exact new config contract in fixtures before judging backend-gate behavior.
- The initial smoke-summary status logic did not infer completion cleanly from trainer state; the implementation had to be patched so the summary reports `12/12` when the raw step metric is absent.

Reusable knowledge:
- Packed-v1 currently needs explicit config gates: `branch_runtime.mode=packed_varlen_exact`, `branch_packing.enabled=true`, `branch_packing.require_flash_attention=true`, `ddp_sync.candidate_padding=none`.
- The packed runtime is still token-cap governed, not memory-budget governed; `memory_target_gib` / `memory_hard_cap_gib` were intentionally left unset/rejected for v1.
- The useful inspection artifact for packed batches is the token-level alignment helper in `src/trainers/stage1_set_continuation/branch_packing.py`.
- The corrected smoke summary now records completion and packed metrics consistently from runtime artifacts.

References:
- [1] `tests/test_stage1_set_continuation_packed_preflight_validation.py` added; passes synthetic packed-vs-serial parity, prior-segment invariance, masked-token loss checks, and alignment-row assertions.
- [2] `src/trainers/stage1_set_continuation/branch_packing.py::packed_alignment_debug_rows(...)` added for token-level inspection.
- [3] `src/sft.py` patched so smoke summary status can derive `12/12` from trainer state when the raw custom metric snapshot omits the field.
- [4] 2-GPU packed smoke completed successfully with `global_step/max_steps: 12/12`, `mp/ddp_candidate_padding_forwards: 0`, and packed telemetry recorded in `effective_runtime.json`.

## Task 2: Diagnose packed-varlen performance and real-model equivalence

Outcome: success

Preference signals:
- The user explicitly asked, “Can we further manage to improve the `pack fill ratio` or other factors to speed up the training while preserving around `60~70GB` memory usage?” which indicates a strong preference for improving compute density without giving back the memory headroom.
- The user asked, “what’s assessment about the `mathematical equivalence` for this new packing mechanism? Same numerical loss?” which indicates that real-model parity, not just synthetic parity, is an important acceptance criterion.
- The user also asked “Did you laugh the failr comparison?” and later requested a prompt for a takeover agent, suggesting they care about explicit fair comparisons and want future agents to be able to continue the work without re-deriving context.

Key steps:
- Ran a packed 2-GPU smoke and a matching smart-batched 2-GPU smoke.
- Compared their throughput and peak memory usage.
- Observed that the current packed implementation used far more memory than smart batching on the normal cap-8 surface and was slightly slower, because the final fill ratio remained low (`~0.174`).
- Determined that the current packed mechanism is stable and structurally validated, but it does **not** yet prove real-Qwen same-batch numerical equivalence.

Failures and how to do differently:
- The early comparison was a fairness/stability smoke comparison, not a frozen same-batch no-step parity harness. Future agents should not call the result “same numerical loss” unless they run a true same-batch parity test with identical weights, branch selection, and no optimizer step.
- The current within-sample packer is structurally limited; trying to tune it harder is unlikely to produce a big fill-ratio gain because it can only pack branches available within one sample.

Reusable knowledge:
- On the current normal-row cap-8 workload, packed varlen is not a throughput win yet: the packed smoke was stable but slower than smart batching and had poor fill.
- Synthetic and structural equivalence evidence is strong, but real-Qwen same-batch numerical equivalence remains unproven.
- A future exactness gate should compare total loss, candidate-balanced loss, close losses, per-candidate scores, and representative gradients on real Qwen weights without an optimizer step.

References:
- [1] Packed smoke artifact: `/data/CoordExp/output_remote/stage1_2b/set_continuation_smoke/packed_varlen_exact_normal_2gpu/smoke-packed-varlen-exact-normal-2gpu/v4-20260427-163412`
- [2] Smart baseline artifact: `/data/CoordExp/output_remote/stage1_2b/set_continuation_smoke/smart_batched_exact_normal_2gpu/smoke-smart-batched-exact-normal-2gpu/v1-20260427-163756`
- [3] Packed smoke summary recorded `completed_max_steps: true`, `train_steps_per_second: 0.104`, peak reserved memory `57.03 GiB`, and `mp/branch_pack_fill_ratio: 0.17414285714285715`.
- [4] Smart smoke summary recorded `completed_max_steps: true`, `train_steps_per_second: 0.113`, peak reserved memory `19.37 GiB`.

## Task 3: Handoff to a new agent for cross-sample branch packing

Outcome: success

Preference signals:
- The user asked for a prompt for another Codex agent to “take over” the new **cross-sample branch packing** in this worktree, indicating they want future agents to continue from the current validated packed-runtime state rather than restart from scratch.
- The user explicitly asked to preserve the current worktree and close the conversation, so the handoff should emphasize continuity, not rework.

Key steps:
- Produced a full takeover prompt for a future agent, including the worktree path, relevant files, validated tests, smoke artifacts, current limitations, and a recommended implementation plan for cross-sample branch packing.
- The prompt instructs the next agent to first build a real-Qwen same-batch parity harness, then implement rank-local cross-sample packing, then add strict equivalence/smoke tests.

Failures and how to do differently:
- Do not infer production readiness from the current packed-vs-smart smoke. The handoff prompt explicitly frames the current packed path as exactness-validated but not yet production-adopted.
- Do not expand the scope to KV-cache or upstream HF model edits; the user’s intent was specifically to improve packed branch execution while preserving objective semantics.

Reusable knowledge:
- The next performance step is likely **rank-local cross-sample branch packing**, not more tuning of within-sample packing.
- Production default should remain `smart_batched_exact` until a workload with denser branch tokens proves the packed path is a throughput win and real-model parity is certified.

References:
- [1] Handoff prompt delivered to the user with explicit instructions for a future Codex agent.
- [2] The handoff prompt names the target direction: `cross-sample branch packing`, `rank-local`, `14000` token cap, `ddp_sync.candidate_padding=none`, and a real-Qwen same-batch parity harness first.

