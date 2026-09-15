---
title: Stage 0 Frozen-Task Pi Harness Screen Results
description: Verified bounded results for the Pi and native Codex frozen-task capability screen.
type: investigation
role: results
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-22-stage0-frozen-task-harness-screen
topic: pi-lightweight-worker-ablation
status: complete
evidence_status: verified_bounded
updated: 2026-07-23
---

# Stage 0 Frozen-Task Pi Harness Screen Results

## Observed

- The user authorized an infrastructure retry after specifying the required
  local proxy at port `9090`. The worktree-local Bash initialization now owns
  uppercase and lowercase proxy variables, and the chroot network preflight
  succeeds before model execution.
- All twelve Pi cells completed with positive token usage and a non-error
  terminal assistant event. All twelve disposable workspaces remained
  byte-for-byte equal to their frozen worker fixtures.
- Luna, Terra, and Sol each passed artifact inventory and deterministic
  aggregation under the original frozen verifiers.
- The original source-routing verifier rejects all three Pi answers because it
  keeps only the last route row per path. The additive audited verifier passes
  Luna and Terra. Sol misspells `HFBackendSession` as `HfBackendSession`; the
  native Sol control made the same error.
- All three Pi answers give the correct claim-audit verdict, qualifying image
  set, and complementarity argument. The original verifier rejects their
  limitations text for missing specific lexical markers. Luna explicitly
  bounds population, training, and architecture claims; Sol and Terra use
  narrower limitations. The strict failures remain recorded.
- The four native Sol controls remain three of four under the original frozen
  verifiers: Tasks 1, 3, and 4 pass, while Task 2 has the exact symbol error.

The complete Pi rerun receipt is:

```text
/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/
2026-07-22-stage0-frozen-task-harness-screen/
pi-proxy9090-rerun1-receipt.md
```

The invalid zero-token attempt remains preserved separately at
`pi-shared-network-failure-receipt.md`. Native answers and verification are
under:

```text
/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/
2026-07-22-stage0-frozen-task-harness-screen/native-controls/
```

## Supported

- Pi `0.81.1` can execute these four bounded read-only task families through
  Luna, Terra, and Sol with full built-in tool access inside a disposable
  chroot while respecting the observable filesystem and command scope.
- Artifact inventory and mechanical aggregation are the strongest immediate
  Pi delegation candidates: every Pi model/reasoning cell passed exactly.
- Luna-high and Terra-xhigh can produce a complete source trace under the
  audited Task 2 semantics. Sol-medium's same error in Pi and native Codex is
  evidence against a Pi-specific explanation for that failure.
- Event-level validation is required. Shell exit status alone hid the original
  transport failure.
- Pi exposes usable per-run token, provider-cost, wall-time, and tool-call
  receipts. Across four cells, reported cost was `0.0595958` for Luna,
  `0.1323005` for Terra, and `0.2385360` for Sol.

## Ruled Out

- The original zero-token failure was not caused by OAuth, Pi model support, or
  the task fixtures; it was caused by clearing the required proxy variables.
- Sol is not automatically the cheapest choice: it used the fewest total
  tokens in this matrix but had the highest provider-reported cost.
- A strict-verifier failure is not automatically a worker-semantic failure;
  the Task 2 row-overwrite defect and Task 3 lexical checks demonstrate that
  verifier design is part of the measured system.

## Unresolved

- Native Codex token and monetary accounting is unavailable, so complete
  delegated-system cost advantage cannot be established.
- The rotated model/reasoning matrix and single observation per cell cannot
  identify a causal model or reasoning-level effect.
- Task 3 needs a frozen semantic verifier or independent adjudication protocol
  before limitations wording can be scored without lexical artifacts.
- Write-heavy work, iterative repair, shared research context, and scientific
  decision ownership remain untested.

## Not Claimed

This unit does not establish that Pi is generally superior to native Codex,
that Luna, Terra, or Sol is globally best, or that provider cost equals total
delegated-system cost. It supports only bounded use for mechanically verifiable
tasks and a larger benchmark before any adapter or default routing promotion.
