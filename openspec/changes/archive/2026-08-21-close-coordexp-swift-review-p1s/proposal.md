# Close 2026-08-21 Post-Archive Review P1s

## Why

The 2026-08-21 Codex adversarial review of the archived four-change program
found three source-verified P1 defects and four P2 candidates. All three P1s
were independently confirmed against source by the Claude lead before this
change was opened. Two P1s are conformance gaps against already-synced stable
specs; one requires the single spec delta in this change.

**Authorization**: this change executes under the user's direct 2026-08-21
approval ("请派遣 subagents,对刚刚与 codex 发现的 issues 做一轮修复,批准"),
ahead of the negotiated Codex-authors-contracts topology taking effect. Codex
remains the intended final acceptor of the fixed tree.

## What Changes

### P1-1 — Cache determinant misses the micro-step assembler (spec conformance)

`src/training/cache_workflow.py::_build_micro_steps_for_dataset` inline-decides
payload-entering fields (metadata composition, `expected_vocab_size`,
`fa2_model_dtype`, `capture_fa2_branch`, `require_fa2_branch_proof`) but is not
in `PACKING_CACHE_DETERMINANT_OWNERS`. The stable spec
(`coordexp-swift-pack-cache-semantic-identity`, "Realized Cache Payload
Identity Is Complete") already requires binding "the source owners that
construct or serialize those payloads" and its scenario names
micro-step-construction owner changes as fingerprint-changing. Fix: extract the
micro-step assembly (the `SupervisedMicroStep` construction loop and its inline
field decisions) into a small dedicated owner module and register it as a new
determinant. Do NOT register the whole 2,140-line `cache_workflow.py` — that
recreates the false-invalidation problem the `cache_contract.py` narrowing
(ca669017c) was built to remove.

**Acceptance evidence flips**: the fingerprint changes BY DESIGN, so
fingerprint equality can no longer be the proof. The proof obligation is
canonical semantic-payload equality on a fixture — identical inputs through
the pre-change and post-change assembler produce the same deterministic
projection of every serialized micro-step field. Raw production pickle chunk
bytes are not a cross-run oracle because PyTorch storage identifiers are not
deterministic; their published digest remains an integrity check for that one
artifact.

**Material consequence (user-acknowledged)**: the next training launch
publishes under a new semantic fingerprint and performs a one-time cache
rebuild. The existing v3 cache (train `8f11237f…` / eval `3b30c157…`) remains
valid evidence for the code that produced it and is not deleted.

### P1-2 — Declared fp16 with missing/disabled GradScaler silently applies (spec delta)

`_active_fp16_scaler` maps a missing or disabled scaler to `None`;
`_reduce_boundary_action` then takes the retained bf16/non-scaler branch and
returns `apply`. Under a declared-fp16 config this silently drops loss-scaling
protection, misdeclares the boundary regime in telemetry, and — under
attribute drift where backward scaled through a scaler the boundary lookup
cannot see (the same shape as the fixed `_scaler_found_inf` defect) — can
apply scaled gradients. Fix, two layers:

1. Uniform launch-time refusal where config and accelerator are both visible,
   before the first training collective: declared fp16 requires an active,
   enabled scaler.
2. Consensus-level backstop: carry declared-fp16 into
   `RankGradientFiniteReport` so `_reduce_boundary_action` refuses the
   all-ranks-scaler-less-under-declared-fp16 state as terminal instead of
   falling through to the bf16 branch. The local report never raises on its
   own — the refusal converges from the gathered all-rank reports so this fix
   cannot itself introduce a pre-collective desync.

The new refusal is a boundary-contract behavior change and is owned by this
change's delta to `coordexp-swift-training-artifacts` (new scenario "Declared
fp16 without an active GradScaler"; all existing scenarios preserved
verbatim).

### P1-3 — Zero-eligible-segment raises locally before the gather (spec conformance)

`_build_denominator_from_token_sequences` raises
`loss.segment_balanced_zero_eligible` locally before
`_resolve_streaming_denominators` gathers, while the stable spec
(`coordexp-swift-supervision-losses`, scenario "Zero eligible protected atoms
on one rank") requires the rank-local eligible count to enter the all-rank
decision before any rank raises, without distributed deadlock. Reachable for
`token_type_gate` on a rank whose shard has no coordinate tokens.

**Pinned failure semantics (not delegated to the builder)**: outcome (a) —
which runs fail does NOT change. Any rank observing zero eligible segments for
a composed term still fails the planned step; the fix makes that failure
collective and clean (every rank enters the gather with its counts, every rank
converges the same typed `loss.segment_balanced_zero_eligible` failure after
the gather). Rescuing local-zero-but-global-nonzero steps would be a
loss-semantics change owned by the user, and is explicitly out of scope. At
`world_size == 1` with no gatherer the existing local raise is retained
unchanged.

### P2 triage (verify first, fix only what is confirmed and cheap)

Each claim is verified against source before any edit; dispositions recorded
in `receipts/p2-triage.md`. Not merge-blocking alongside the P1s:

1. Terminal row loses known finite truth.
2. `TrainingExecutionPlan` is only shallowly immutable.
3. Phase finalization has dual owners.
4. Import-boundary guard misses `from package import member`.

## Non-Goals

- No loss-semantics change (zero-eligible failing runs still fail).
- No architecture compression of `session.py` / `cache_workflow.py` beyond the
  small assembler extraction P1-1 requires.
- No cache rebuild is executed by this change; the rebuild happens at the next
  authorized training launch.
- The four P2s do not block completion if triage dispositions record them as
  deferred with an owner.

## Impact

- Affected specs: `coordexp-swift-training-artifacts` (one MODIFIED
  requirement, one added scenario).
- Affected code: `src/losses/runner.py`, `src/runtime/train_runtime.py`,
  `src/runtime/finite_gates.py`, `src/training/cache_workflow.py`,
  `src/training/pack_cache.py`, one new assembler owner module, tests.
- Fingerprint identity: future fingerprints change (new determinant); the
  canonical semantic payload projection is unchanged.
