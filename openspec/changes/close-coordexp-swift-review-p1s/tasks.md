# Tasks — close-coordexp-swift-review-p1s

Budget envelope (negotiated 2026-08-21): RED-first receipts per builder, one
bundled correction round each, no per-builder Opus audits, full suite at most
once, Codex is the final acceptor of the fixed tree.

## Wave 1 — parallel builders (disjoint file ownership)

- [x] 1.1 Builder A (owns `src/losses/runner.py` + its tests): P1-3 collective
      zero-eligible decision per pinned semantics (a); RED test first
      reproducing the pre-gather local raise at ws>1, then fix; ws==1 local
      raise retained and tested.
- [x] 1.2 Builder B (owns `src/runtime/train_runtime.py`,
      `src/runtime/finite_gates.py` + their tests): P1-2 two-layer fail-closed
      per delta scenario; RED test first showing declared-fp16 + no scaler
      currently converges `apply`.
- [x] 1.3 Builder C (owns `src/training/cache_workflow.py`,
      `src/training/pack_cache.py`, new assembler module + tests): P1-1
      assembler extraction + new determinant registration; payload-byte
      equality fixture proof; `_DETERMINANT_REASONS` parity.
- [x] 1.4 Builder D (verify-only): source-verify the four P2 claims, write
      dispositions to `receipts/p2-triage.md`; NO edits in this wave.

## Wave 2 — integration and verification (lead)

- [x] 2.1 Integrate A/B/C; apply D's confirmed-and-cheap fixes (terminal-row
      P2 lands only after B, same files).
- [x] 2.2 Timeout-bounded 2-rank gloo probe: zero-eligible rank converges the
      same typed failure on both ranks within the bound (no hang).
- [x] 2.3 fp16 fail-closed probe: declared fp16 + absent/disabled scaler
      refuses at launch; consensus backstop covered by unit tests.
- [x] 2.4 One full CPU suite run; judge by failure-set diff against the known
      baseline (126 historicized-executor skips expected).
- [x] 2.5 Commit as one revertible commit with `git apply -R --check` proof.

## Wave 3 — close-out

- [x] 3.1 Scenario-diff pre-check on the delta (archive fail-close trap),
      `openspec validate --all`.
- [ ] 3.2 Close-out note + command manifest; hand the tree to Codex for final
      acceptance; archive only after that acceptance.

## Wave 2/3 close-out (2026-08-21, lead)

- Wave 1: four builders delivered with RED receipts (A: pre-gather raise proven
  then collectivized; B: two-layer fp16 fail-closed with three REDs; C:
  characterization green-before/green-after + sensitivity, payload bytes
  byte-identical across the extraction; D: four P2 verdicts with file:line
  evidence).
- Wave 2: P2-1/P2-2/P2-4 fixed (dispositions + the rejected MappingProxyType
  variant recorded in receipts/p2-triage.md); P2-3 DEFERRED (resume-semantics
  decision, user-owned). Probes 2.2/2.3 PASS with pre-fix simulations showing
  the real deadlock and the real silent fp16 convergence. Full suite 2600/126skip
  EXIT=0, failure-set diff vs baseline EMPTY, flake tripwire untouched (2/3).
- Determinant registry: +1 owner (micro_step_assembler); future fingerprints
  change BY DESIGN; next training launch performs a one-time cache rebuild.
  WAVE0_DETERMINANT_OWNERS amended in place following that file's Wave-3
  precedent (inline comment names this change) — flagged for the acceptor.
- Declared behavior change (P1-2, disclose to acceptor): a config declaring
  fp16 on a host/path where no enabled GradScaler resolves — including
  CUDA-less CPU launches that previously "ran" fp16 unprotected — now refuses
  at launch with `runtime.fp16_scaler_missing`. This is the delta scenario's
  intent, not a side effect.
- Accepted-risk dispositions (P1-2, lead): (a) mixed declared_fp16 across
  ranks with zero scaler ranks still takes the retained branch — accepted as
  unreachable-by-construction (config is identical on every rank; the launch
  gate owns that state); (b) scaler-drift end-to-end replay not exercised —
  the consensus backstop is proven at the `_reduce_boundary_action` seam per
  task 2.3's own acceptance wording (unit-covered backstop).
- Wave-1 RED evidence is durable in receipts/wave-1-builder-red-receipts.md
  (verbatim pre-fix failures for A/B/C).
- 3.2 remains open: the tree is handed to Codex for final acceptance; archive
  only after that acceptance.
