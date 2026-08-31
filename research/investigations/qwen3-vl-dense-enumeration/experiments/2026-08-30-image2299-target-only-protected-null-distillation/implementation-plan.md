# Target-only protected-null distillation implementation plan

## Execution contract

Use the existing target-only protected-null runner and its frozen receipt
contract. Bind the source receipt, model/image/prompt surfaces, protected-null
identity, route hashes, solver limits, and generic matcher gate. Run staged
constraints in order, stopping at the first failed ordinary-greedy gate.

## Completion check

- [x] Independently verified the v1 receipt SHA-256.
- [x] Stage 1 exact route and gate passed.
- [x] Stage 2 solver and full-vocabulary runtime checks passed, but its
      ordinary-greedy route failed the exact expected-length and debt gate.
- [x] Stopped at stage 2; stages 3--5 remain unexecuted.
- [x] Recorded the bounded interpretation and one-shot 108-constraint
      successor without executing it.

No scripts, tests, model weights, checkpoints, or runtime outputs were changed
by this documentation seal.
