## Context

CoordExp Stage-2 training uses multi-process DDP. The most failure-prone pattern in DDP is:
- rank-local exception handling that changes control flow (skip/continue/return),
- followed by a distributed collective on other ranks,
→ resulting in a deadlock/hang rather than a clear exception.

We already encountered and fixed a Stage-2 AB DDP deadlock by aligning a final backward/sync boundary across ranks. This change turns that “lesson learned” into a systematic contract.

Constraints:
- Config-first; avoid new ad-hoc CLI flags.
- Preserve existing geometry and packing invariants (no coordinate reorder/drop).
- Do not modify upstream HF model files.
- Keep fixes minimal and evidence-backed; no broad refactors.

## Goals / Non-Goals

**Goals**
- Enforce strict fail-fast semantics for DDP-critical regions:
  - no swallow-and-continue,
  - no “local fallback” when collectives are expected,
  - coordinated termination so all ranks fail together (avoid deadlocks).
- Make the “safe rank0-only side effect under DDP” pattern canonical and reusable.
- Ensure “timeout-based waiting” logic is actually bounded (no single probe blocks forever).

**Non-Goals**
- No global exception policy allowlists/registries.
- No new distributed runtime framework beyond small, local helpers.
- No attempt to make upstream libraries (vLLM/ms-swift/accelerate) fully strict; focus on CoordExp’s integration boundaries and safe defaults.

## Decisions

1) **DDP-critical regions are always strict**
- If `torch.distributed` is initialized (`world_size > 1`), any unexpected exception in a region that can reach a collective MUST terminate the run.
- “Proceed with local metrics” (or any local fallback) is forbidden once a collective is expected; it is both correctness-risky and deadlock-prone.

2) **Canonical coordinated-failure patterns**

2a) **Rank0-only side effects (bounded; no raise-before-broadcast)**
For any rank0-only operation that must be synchronized, all ranks MUST execute the following **bounded** pattern:
- bounded barrier (align entry),
- rank0 executes the side effect inside `try/except`, capturing `{failed_flag, error_summary}`,
- rank0 broadcasts `{failed_flag, error_summary}` to all ranks,
- bounded barrier (align exit),
- only after broadcast + exit alignment completes, raise on all ranks if `failed_flag` is set (rank0 logs full traceback).

Rank0 MUST NOT raise (or return) before completing the broadcast and exit alignment barrier; exceptions are caught only to coordinate termination and are then re-raised.

If the bounded barrier mechanism (e.g., monitored barrier / monitor group) is enabled/configured but cannot be initialized, the system MUST fail fast with actionable guidance and MUST NOT silently downgrade to an unbounded barrier.

2b) **Any-rank exceptions (pre-collective) inside DDP-critical regions**
If a DDP-critical operation is executed by all ranks and may raise **before entering the next distributed collective**, implementations MAY catch exceptions **only** to coordinate a rank-symmetric termination:
- execute the operation inside a local `try/except` on each rank (capture local failure flag + short summary),
- perform a rank-symmetric failure coordination step (e.g., reduce a tensor failure flag so all ranks agree on “any failure happened”),
- raise on all ranks if any failure occurred.

This preserves rank symmetry and prevents hangs where “rank k fails, rank j continues into the next collective”.

**Important:** this pattern MUST NOT wrap `torch.distributed` collectives. If a distributed collective itself raises (or indicates process-group corruption), do not attempt additional coordination collectives; re-raise and let the distributed backend error/timeout rather than risking a second deadlock.

3) **No “best-effort” wrappers around collectives**
Diagnostics-only wrappers (e.g., “warn once and disable diagnostic”) are allowed only when:
- they do not call `torch.distributed` collectives, and
- disabling the diagnostic cannot change rank-symmetric behavior.

If a diagnostic requires collectives under DDP, it must either:
- be made rank-symmetric and strict (fail-fast), or
- be disabled globally in a synchronized way.

4) **Timeout-only readiness semantics (launcher)**
For server readiness, keep simple “wait until overall timeout” behavior, but enforce:
- each probe call is bounded (connect + total time),
- probe failures are treated as “not ready yet” until `WAIT_TIMEOUT` is exceeded.

No new early-abort rules are introduced beyond existing hard-error guards (e.g., invalid config, port already in use).

## Risks / Trade-offs

- More runs will terminate earlier (intentional): this is preferable to silent divergence or deadlocks.
- Some “metrics-only” features may become strict in DDP; this is acceptable since metric aggregation uses distributed collectives and is therefore DDP-critical.
- Bounded probe timeouts may surface flaky networking sooner; operators can adjust `WAIT_TIMEOUT` rather than relying on unbounded hangs.
