# Canonical-G46 global-QP sentinel implementation plan

## Execution truth

Complete. The v3 runner froze 57 positive positions, 51 selected rows, rank 57,
and one 2,907-variable/2,907-constraint QP. One certified finite primal entered
one warm decode and one fresh-cold replay; both passed exact G46. No route,
surface, norm cap, iteration, second solve, or composed parent payload was used.

1. Add one dedicated runner and focused CPU invariant test; do not modify any
   completed runner or receipt.
2. Reuse existing target loading, route hashing, production parser/matcher,
   hidden capture, protected-null basis, target-only minimum-normalized solver,
   full-vocabulary check, sparse output residual, surface sentinels, and cold
   subprocess helpers.
3. `--check-bindings` must reproduce the exact 415-token route and static 46/46
   gate without model/GPU loading.
4. The GPU run captures dynamic positives, freezes `S`, solves exactly once,
   records rank/variables/constraints/minimum norm/slack, and runs at most one
   warm candidate. Only warm exact success writes a payload and enters fresh
   cold verification.
5. Persist a typed receipt for success, scientific negative, infeasibility, or
   technical HOLD. Preserve the first divergence for any nonexact decode.
