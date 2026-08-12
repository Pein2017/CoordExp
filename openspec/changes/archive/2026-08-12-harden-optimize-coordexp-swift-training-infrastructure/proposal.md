## Why

The accepted CoordExp-Swift training path is accurate on its current packed
Qwen3-VL/FA2 route, but a focused 2026-08-06 audit found three gaps that block
stronger production claims: packing-cache identity can silently miss tokenizer
or cached-payload-owner changes, FA2 proof does not attest every executed text
layer, and a zero-weight token-type diagnostic still builds an expensive
full-vocabulary autograd graph. The same audit also found bounded opportunities
in cache admission, eval hydration, CPU/GPU input overlap, packing utilization,
run provenance, and exact training-state resume that should be handled through
separate measured waves rather than one coupled rewrite.

## What Changes

- Make packing-cache identity complete for every transitive producer of cached
  micro-step state, including full tokenizer/processor assets, realized
  vocabulary groups, raw-data parsing, and cache-construction code. Publish v3
  only under a previously absent version namespace/fingerprint and reject
  semantically stale caches before model loading without replacing an old root.
- Strengthen packed FA2 attestation from “one matching varlen call” to a proof
  that every executed Qwen text layer used the configured backend and the
  exact explicit boundaries; retain packed-versus-unpacked forward/backward
  parity as the numerical acceptance oracle. The accepted Wave 2 v3 oracle
  keeps the Qwen attention/linear path under BF16 autocast, preserves the
  production graph-connected `ConvertOutputsToFp32` logits/loss seam, detaches
  only receipt/comparison snapshots, uses one elementwise tolerance for
  BF16-compute-derived values, and reproduces production streaming backward
  cadence for the separate reference.
- Preserve zero-weight loss diagnostics without retaining their gradient graph;
  require exact diagnostic equivalence and production-shaped memory/wall-clock
  evidence before accepting the optimized path.
- Move cache admission ahead of model/adaptor allocation, load disjoint eval
  shards directly instead of hydrating the full eval cache on every rank, and
  add phase/RSS/I/O receipts that distinguish preparation, publication,
  admission, model load, first optimizer step, and steady state.
- Re-evaluate three actually distinct preparation paths on the exact eight-rank
  production shape: the legacy fused device-direct build, the current
  synchronous CPU-build-then-transfer provider, and the bounded depth-one
  overlapped provider. Promote a new default only after a reproducible
  end-to-end win with identical input, loss, optimizer, and artifact streams;
  the lead owns the measured default disposition.
- Add explicit, fingerprinted packing policies so source-order next-fit remains
  the compatibility default while fixed-window binpacking and bounded online
  packing can reorder atomic image examples across packs or batches. Packing
  MUST preserve the fixed object/row order inside every image under the resolved
  sorted policy and MUST make cross-image order/co-presentation drift explicit.
- Add exact training-state resume for optimizer, scheduler, mixed-precision
  state when applicable, accumulation/data cursor, RNG state, pack/cache
  identity, and artifact continuation. The first contract is opt-in,
  same-world-size, optimizer-step-boundary only, saves every trainable surface
  while binding rather than copying frozen base weights, and does not
  automatically prune exact-state checkpoints. Model-only checkpoints remain
  valid for inference but MUST NOT be presented as exact training resume.
- Extend the compact run/environment receipt with repository commit/diff state,
  installed Transformers source identity, FlashAttention binary provenance,
  and Torch/Accelerate/PEFT/tokenizers versions. This change keeps the installed
  dependency and FA2 backend baseline; upgrades require a future isolated
  change rather than an incidental comparison arm here.
- Organize implementation as dependency-ordered waves with a correctness gate,
  a measurement gate, and an independent audit at every claim-critical
  boundary. A failed performance gate reverts only that candidate and does not
  block correctness hardening or unrelated accepted waves.

## Capabilities

### New Capabilities

- `coordexp-swift-training-resume`: Exact, fail-closed continuation of training
  state and artifact lineage across an interrupted run.

### Modified Capabilities

- `coordexp-swift-pack-cache-semantic-identity`: Bind every realized cached
  payload and its transitive producers; version and reject old identities; make
  admission happen before expensive distributed model setup.
- `coordexp-swift-packing-forward`: Strengthen all-layer FA2 execution proof and
  define explicit, receipt-bearing packing-policy alternatives without changing
  the compatibility default silently.
- `coordexp-swift-supervision-losses`: Preserve zero-weight term diagnostics
  while prohibiting unnecessary gradient graphs and proving numerical
  equivalence.
- `coordexp-swift-training-artifacts`: Record exact code/dependency provenance,
  startup/steady-state phase measurements, cache/policy identity, and resume
  lineage in additive artifacts.
- `coordexp-swift-config-runtime`: Add fail-fast controls for packing policy,
  input-provider promotion, resume mode, and upstream-backend compatibility.

## Protected Semantics And Non-Goals

- Do not change renderer text, tokenization, object or coordinate meaning,
  supervision placement, segment-balanced loss normalization, DDP scaling,
  Qwen visual replacement, four-row per-segment MRoPE, or explicit FA varlen
  boundaries except through an independently approved research change.
- Treat one encoded image example as the atomic packing unit. Under a sorted
  policy, the resolved object/row order inside that image MUST remain fixed.
  Whole-image examples MAY be reordered across packs or batches and may receive
  different co-present examples when the policy is deterministic, fingerprinted,
  each-once, and passes the named correctness and efficiency gates.
- Do not enable KV cache for full-sequence training and do not cache GPU vision
  features, hidden states, or trainable-tower prefixes in this change.
- Do not make binpacking, online packing, overlapped input preparation, FA3/FA4,
  or a new dependency version the production default merely because the code
  path exists. Each promotion requires its named equivalence and wall-clock
  gate.
- Do not delete, mutate, or replace an existing production-cache directory.
  In this change, `rebuild` means only publishing a newly resolved cache under
  a previously absent v3 namespace/fingerprint. Cache cleanup is out of scope.
- Do not claim bitwise reproducibility across independent CUDA launches; exact
  resume requires state/cursor continuity and declared numerical tolerances,
  not an unsupported cross-launch bitwise guarantee.

## Resolved Owner Decisions

The user resolved the prior decision gates on 2026-08-06:

1. the lead owns input-provider and packing-policy promotion after the frozen
   semantic, resource, and end-to-end wall-clock gates;
2. cross-image and cross-batch reordering/co-presentation is admissible, while
   object/row order inside one image under the sorted policy remains protected;
3. the lead owns the exact-resume design under the bounded first-version
   contract stated above;
4. no dependency or attention-backend upgrade is in scope for this change;
5. the one production cache-version materialization and final multi-GPU
   convergence run are authorized after their existing preconditions pass.

On 2026-08-10 the user also authorized one conditional Wave 2 v3 reopening.
The immutable v2 failure remains historical evidence and is not reinterpreted.
V3 keeps the exact v2 workload and installed dependency baseline, uses one
same-packed repeat only as a fixed measurability check under plan-bound,
probe-only `FLASH_ATTENTION_DETERMINISTIC=1`, and permits at most one
new real-model GPU execution after the v3 contract, CPU tests, exact launch
packet, and independent P0/P1 audits pass. There is no automatic retry, sample
switch, tolerance change, or v4 continuation.

After the immutable v3 attempt completed, the user selected a narrow release
disposition rather than a retry or a wider threshold. The accepted Wave 2 claim
is limited to the observed byte-identical supervised forward, matching
loss/denominator semantics, decisive boundary negative, and exact 28-layer FA2
execution proof. The packed-versus-streaming cross-shape gradient result remains
an immutable failed diagnostic and is not an acceptance gate for Wave 3. This
does not claim exact Jacobian equality or authorize another Wave 2 execution.

On 2026-08-11 the user additionally authorized correctness and plumbing smoke
work to share the eight GPUs with already-running jobs.  This permits a bounded,
stable pre-existing compute-process baseline for r5 determinism and exact-resume
execution, but it does not convert shared-load wall time, utilization, or memory
observations into provider, packing, startup, zero-weight, or throughput
promotion evidence.  The bounded admission allows at most 49152 MiB of
pre-existing allocation per 81920 MiB GPU, retaining at least 32768 MiB for the
owned smoke.  Controllers must preserve and never signal the baseline
jobs, reject any newly appearing or surviving owned GPU process after a phase,
and retain the existing no-retry and cost limits.

Fresh versioned Wave 3 through Wave 6 successors may use that same availability
contract for numerical, artifact, correctness, and plumbing evidence only. They
must use new absent roots and one-shot markers; historical or consumed roots are
never reactivated. All provider, packing, startup, throughput, timing, memory,
and efficiency promotion claims continue to require matched, otherwise-idle
evidence.

Later on 2026-08-11 the user replaced that optional successor route with the
shortest production-trust path: finish Wave 7, then Wave 8, then Wave 9. No new
Wave 3, Wave 4, or Wave 5 performance marker will be consumed. Wave 3 closes
without an efficiency claim, Wave 4 retains the current correctness-tested eval
hydration behavior without a startup/resource promotion claim, and Wave 5
retains the synchronous provider default without a provider-performance claim.
Wave 6 remains explicitly pending for future design and matched-training
research; `source_order_next_fit` remains the production default, and that
pending work does not block the current Wave 7 -> Wave 8 -> Wave 9 convergence.
This is a disposition decision, not retroactive evidence that an omitted
experiment passed.

No further user decision is required unless implementation would change the
protected intra-image order, exact-resume compatibility boundary, dependency
baseline, declared cost envelope, or another research-semantic promise.

The completed `streamline-coordexp-swift-base-infrastructure` change was synced
to stable specs and archived at
`openspec/changes/archive/2026-08-06-streamline-coordexp-swift-base-infrastructure/`,
so its accepted M2/M4/M6 deltas are now the stable baseline rather than a
competing active authority.

## Impact

- Primary code owners: `src/training/pack_cache.py`,
  `src/training/pipeline.py`, `src/training/forward_input_provider.py`,
  `src/training/supervised_trainer.py`, `src/qwen/fa2.py`,
  `src/qwen/forward.py`, `src/losses/runner.py`, `src/losses/vocab.py`,
  `src/eval/forward.py`, config models, checkpoint/artifact writers, and
  focused tests.
- Caches: one new v3 production fingerprint after correctness owners settle;
  publication never replaces a path, and old caches remain immutable historical
  acceleration state.
- Artifacts: additive run/environment, packing-policy, phase-timing, FA proof,
  and exact-resume fields; existing model-only inference checkpoints remain
  readable under their current contract.
- Dependencies: no version upgrade in this change. Installed
  `ms-swift==4.2.2` and `transformers==4.57.1` are comparison/provenance
  anchors, not new runtime ownership for CoordExp packing or losses.
- Cost: CPU-only/unit waves first; bounded one-GPU parity probes next;
  authorized eight-rank performance and resume interruption probes only after
  their preconditions pass and within the frozen cost envelope.
